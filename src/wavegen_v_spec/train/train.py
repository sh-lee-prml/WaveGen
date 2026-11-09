"""Full-generator VLoss + SF + MS-V-Spec0.1 tuning, with no GAN components.

Fresh Online weights come only from the original1300k EMA. All generator
parameters train, including waveform and Mel/SF branches. Train three prepared
loader epochs with fresh AdamW, warmup2000 and constant LR2e-6. A fresh native
EMA starts with an exact Online copy after successful stage update2000.
This entrypoint never constructs a discriminator or an adversarial/FM objective.
"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Mapping

from importlib.resources import files

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from wavegen_v_spec.model.cfm import CFMMSVSpec
from wavegen_v_spec.train.delayed_ema import (
    DelayedGeneratorEMA, EMA_CONTRACT, EMA_START_UPDATE, EMA_DECAY, EMA_UPDATE_EVERY,
)
from wavegen_v_spec.model.dataset import DynamicBatchSampler, collate_fn, load_dataset
from wavegen_v_spec.model.utils import get_tokenizer
from wavegen_v_spec.train.checkpoint import (
    CheckpointIO, relocated_stage_source,
    REPO_ROOT, SOURCE_UPDATE, SOURCE_CHECKPOINT_SIZE, SOURCE_CHECKPOINT_SHA256,
    SOURCE_VOCAB_SHA256, BASE_CHECKPOINT_KEYS, STAGE_FORMAT,
    SPECTRAL_CONTRACT as _PREVIOUS_SPECTRAL_CONTRACT,
    _load_pt_checkpoint, _plain_dict, _resolve_path, _strip_source_aux_mel_buffers,
    _validate_optimizer_state,
)

CONFIG_NAME = "WaveGen_VSpec"
OUTPUT_NAME = "WaveGen_VSpec_Train"
STAGE_KEY = "full_msvspec01_nogan_ema2000_3ep_stage"
RNG_STATE_KEY = "full_msvspec01_nogan_ema2000_3ep_rng_states"
STAGE_CHECKPOINT_KEYS = BASE_CHECKPOINT_KEYS | {STAGE_KEY, RNG_STATE_KEY}
STAGE_EPOCHS = 3
TRAINING_HORIZON_CONTRACT = {"unit": "prepared_loader_epochs", "stage_epochs": STAGE_EPOCHS,
                             "updates_per_epoch": "len(prepared_loader)",
                             "gradient_accumulation_steps": 1}
TARGET_LEARNING_RATE = 2e-6
WARMUP_UPDATES = 2000
WARMUP_START_LR = 2e-7
LR_SCHEDULE_CONTRACT = {
    "type": "linear_warmup_then_constant", "num_warmup_updates": WARMUP_UPDATES,
    "start_lr": WARMUP_START_LR, "target_lr": TARGET_LEARNING_RATE,
    "counter": "completed_stage_optimizer_updates", "warmup_included_in_total": True,
}
SPECTRAL_CONTRACT = {
    **_PREVIOUS_SPECTRAL_CONTRACT, "weight": 0.1,
    "time_scaling": "none", "resolution_reduction": "seven_scale_sum",
}
LOSS_CONTRACT = {
    "prediction": "x_pred", "primary_loss": "masked_waveform_v_MSE",
    "sf_loss_weight": 0.1, "spectral": SPECTRAL_CONTRACT,
    "aux_mel": False, "repa": False, "perceptual": False, "gan": False,
    "head_training_policy": "all_generator_parameters_train_including_waveform_and_Mel_DDT",
    "discriminator": False, "adversarial": False, "feature_matching": False,
    "data_normalization_added": False,
}


def learning_rate_at_update(completed_updates: int) -> float:
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("completed_updates must be a nonnegative integer")
    fraction = min(completed_updates / WARMUP_UPDATES, 1.0)
    return WARMUP_START_LR + fraction * (TARGET_LEARNING_RATE - WARMUP_START_LR)


def _validate_config(cfg: DictConfig) -> int:
    """Keep the numerical recipe fixed; allow explicit filesystem/logging choices."""
    reference = OmegaConf.load(str(files("wavegen_v_spec").joinpath("configs/WaveGen_VSpec.yaml")))
    for section in ("model", "datasets", "optim", "ema"):
        if _plain_dict(cfg[section]) != _plain_dict(reference[section]):
            raise ValueError(f"Preserve the registered V-Spec {section} settings")
    for key in ("stage_epochs", "num_warmup_updates", "grad_accumulation_steps"):
        if type(cfg.optim[key]) is not int:
            raise ValueError(f"optim.{key} must be an integer")
    if type(cfg.ema.start_update) is not int or type(cfg.ema.update_every) is not int:
        raise ValueError("EMA update counters must be integers")
    expected = _plain_dict(reference.ckpts)
    for key in ("source_ckpt", "save_dir", "logger", "resume_ckpt"):
        expected[key] = cfg.ckpts[key]
    if _plain_dict(cfg.ckpts) != expected:
        raise ValueError("Preserve the checkpoint identity and 10000-update save intervals")
    if _resolve_path(cfg.ckpts.save_dir, name="save_dir").name != OUTPUT_NAME:
        raise ValueError(f"Training output must end in {OUTPUT_NAME}")
    return 512


def _validate_model_contract(model, *, mask_align: int = 512) -> None:
    if type(model) is not CFMMSVSpec:
        raise ValueError("Use unchanged CFMMSVSpec, without REPA or encoder-loss extensions")
    if model.aux_mel_loss is not None or model.use_aux_mel_loss:
        raise ValueError("AuxMel must remain disabled")
    if not model.use_ms_v_spec_loss or model.ms_v_spec_loss_weight != 0.1 or model.sf_loss_weight != 0.1:
        raise ValueError("Use waveform v-loss + SF0.1 + MS-V-Spec0.1")
    if model.prediction != "x_pred" or model.loss_space != "v" or model.mask_align_to != mask_align:
        raise ValueError("Preserve original prediction, v-loss and mask alignment")
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("All generator parameters, including both DDT branches, must train")


def _require_exact_tensor_state(state, target, *, name):
    if not isinstance(state, Mapping) or set(state) != set(target):
        raise RuntimeError(f"{name}: tensor-state keys differ")
    for key, reference in target.items():
        value = state[key]
        if not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise RuntimeError(f"{name}: tensor shape/dtype differs: {key}")


def source_ema_initial_state(source_ema, target_online):
    """Read ONLY source EMA weights; source Online is deliberately not an input."""
    if int(source_ema["step"].item()) != SOURCE_UPDATE or not bool(source_ema["initted"].item()):
        raise RuntimeError("Expected initialized source EMA at1300000")
    source = _strip_source_aux_mel_buffers(source_ema, ema=True)
    expected = {"step", "initted"} | {"ema_model." + key for key in target_online}
    if set(source) != expected:
        raise RuntimeError("Source EMA contains missing or foreign model keys")
    online = {key: source["ema_model." + key] for key in target_online}
    _require_exact_tensor_state(online, target_online, name="source EMA -> Online")
    return online


class VSpecTrainer(CheckpointIO):
    """Full-generator three-epoch loop with fresh delayed native EMA."""

    def __init__(self, model, *, source_ckpt: Path, training_contract: Mapping,
                 checkpoint_path, resume_checkpoint=None, stage_epochs: int = STAGE_EPOCHS,
                 learning_rate=TARGET_LEARNING_RATE, num_warmup_updates=WARMUP_UPDATES,
                 batch_size_per_gpu=19200, batch_size_type="frame", max_samples=64,
                 grad_accumulation_steps=1, max_grad_norm=1.0,
                 save_per_updates=10000, last_per_updates=10000, keep_last_n_checkpoints=-1,
                 logger="wandb", wandb_project="WaveGen", wandb_run_name=OUTPUT_NAME,
                 log_samples=False, bnb_optimizer=False, model_cfg_dict=None):
        self.source_ckpt = Path(source_ckpt).resolve()
        self.resume_checkpoint = (Path(resume_checkpoint).resolve() if resume_checkpoint else None)
        self._saved_source_path = None
        self.start_update = SOURCE_UPDATE
        self.training_contract = dict(training_contract)
        if type(stage_epochs) is not int or stage_epochs != STAGE_EPOCHS:
            raise ValueError("Train exactly three prepared-loader epochs")
        self.stage_epochs, self.total_updates = stage_epochs, 0
        if (save_per_updates, last_per_updates, keep_last_n_checkpoints) != (10000, 10000, -1):
            raise ValueError("Save every10000 updates and at completion, without deleting old checkpoints")
        output = Path(checkpoint_path).resolve()
        if output == self.source_ckpt.parent or output in self.source_ckpt.parents or self.source_ckpt.parent in output.parents:
            raise ValueError("Output must be separate from the source checkpoint directory")
        if self.source_ckpt.suffix != ".pt" or self.source_ckpt.name.endswith("_ema.pt"):
            raise ValueError("Use the original full1300k checkpoint")
        _validate_model_contract(model)
        if (grad_accumulation_steps, num_warmup_updates, learning_rate) != (1, WARMUP_UPDATES, TARGET_LEARNING_RATE):
            raise ValueError("Use accumulation1,2000 warmup updates and2e-6 target LR")
        if bnb_optimizer or log_samples or batch_size_type != "frame":
            raise ValueError("Use plain AdamW/frame batching without sample logging")
        if logger == "wandb":
            import wandb
            if not wandb.api.api_key:
                logger = None
        # Do not call base Trainer.__init__: it would allocate a full EMA copy.
        self.accelerator = Accelerator(
            log_with="wandb" if logger == "wandb" else None,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            gradient_accumulation_steps=1)
        if self.accelerator.num_processes != 8 or self.accelerator.mixed_precision != "bf16":
            raise RuntimeError("Preserve the8-process BF16 geometry")
        self.logger, self.log_samples = logger, False
        if logger == "wandb":
            self.accelerator.init_trackers(wandb_project, config=model_cfg_dict or {},
                init_kwargs={"wandb": {"resume": "allow", "name": wandb_run_name}})
        elif logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")
        self.checkpoint_path = str(output)
        self.batch_size_per_gpu, self.batch_size_type = batch_size_per_gpu, batch_size_type
        self.max_samples, self.max_grad_norm = max_samples, max_grad_norm
        self.grad_accumulation_steps, self.num_warmup_updates = 1, WARMUP_UPDATES
        self.save_per_updates, self.last_per_updates = save_per_updates, last_per_updates
        self.keep_last_n_checkpoints, self.noise_scheduler = keep_last_n_checkpoints, None
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                           betas=(.9, .999), eps=1e-8, weight_decay=.01)
        self.model, self.optimizer = self.accelerator.prepare(model, self.optimizer)
        self.ema = DelayedGeneratorEMA(self.accelerator.unwrap_model(self.model),
                                       enabled=bool(self.accelerator.is_main_process))
        self._stage_update = self._stage_epoch = self._next_batch_in_epoch = 0
        self._prepared_batches_per_epoch = 0
        self._source_checkpoint_size = None
        self._resumable_seed = None
        self._pending_rng_state = None
        self._set_stage_learning_rate()

    def _set_stage_learning_rate(self):
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate_at_update(self._stage_update)

    def _lr_schedule_state(self):
        return {**LR_SCHEDULE_CONTRACT, "completed_updates": self._stage_update,
                "next_lr": learning_rate_at_update(self._stage_update)}

    def _expected_stage_contract(self):
        return {
            "format": STAGE_FORMAT, "version": 1,
            "source_checkpoint": self._saved_source_path or str(self.source_ckpt), "source_update": SOURCE_UPDATE,
            "source_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_state_policy": "source_EMA_to_Online_only__source_Online_unused",
            "source_buffer_migration": "remove_exact14_source_EMA_aux_mel_buffers_in_memory",
            "ema_tracking": True, "delayed_generator_ema": EMA_CONTRACT,
            "trainable_policy": "all_generator_parameters",
            "DDP_find_unused_parameters": True,
            "training_horizon": TRAINING_HORIZON_CONTRACT,
            "total_optimizer_updates_including_warmup": self.total_updates,
            "lr_schedule": LR_SCHEDULE_CONTRACT, "objective": LOSS_CONTRACT,
            "prepared_batches_per_epoch": self._prepared_batches_per_epoch,
            "num_processes": self.accelerator.num_processes,
            "mixed_precision": self.accelerator.mixed_precision,
            "resumable_seed": self._resumable_seed,
            "training_contract": self.training_contract,
        }

    def _stage_metadata(self):
        return {"contract": self._expected_stage_contract(),
                "source_checkpoint_size": self._source_checkpoint_size,
                "stage_epoch": self._stage_epoch, "next_batch_in_epoch": self._next_batch_in_epoch,
                "stage_update": self._stage_update, "ema_tracking": self.ema.tracking_state()}

    def _validate_stage_cursor(self, checkpoint, stage):
        expected_keys = {"contract", "source_checkpoint_size", "stage_epoch", "next_batch_in_epoch", "stage_update", "ema_tracking"}
        if not isinstance(stage, Mapping) or set(stage) != expected_keys:
            raise RuntimeError("Missing or foreign full-model NoGAN stage metadata")
        self._saved_source_path = relocated_stage_source(
            stage["contract"], self._expected_stage_contract(), stage["source_checkpoint_size"])
        epoch, cursor, update = (stage[k] for k in ("stage_epoch", "next_batch_in_epoch", "stage_update"))
        if any(type(v) is not int for v in (epoch, cursor, update)):
            raise RuntimeError("Stage cursor must contain integers")
        if not (0 <= epoch <= self.stage_epochs and 0 <= cursor < self._prepared_batches_per_epoch
                and 0 <= update <= self.total_updates):
            raise RuntimeError("Stage cursor exceeds its data/update budget")
        if (update != epoch * self._prepared_batches_per_epoch + cursor
                or type(checkpoint["update"]) is not int
                or checkpoint["update"] != SOURCE_UPDATE + update):
            raise RuntimeError("Stage cursor and absolute update disagree")
        self._stage_epoch, self._next_batch_in_epoch, self._stage_update = epoch, cursor, update
        self._source_checkpoint_size = stage["source_checkpoint_size"]
        if checkpoint["scheduler_state_dict"] != self._lr_schedule_state():
            raise RuntimeError("Saved warmup state differs from stage updates")

    def _set_pending_rng_state(self, checkpoint):
        states = checkpoint[RNG_STATE_KEY]
        if not isinstance(states, list) or len(states) != self.accelerator.num_processes:
            raise RuntimeError("Stage needs exactly one RNG state per rank")
        matching = [s for s in states if isinstance(s, Mapping) and s.get("rank") == self.accelerator.process_index]
        if len(matching) != 1:
            raise RuntimeError("Missing unique RNG state for this rank")
        self._pending_rng_state = matching[0]

    def load_checkpoint(self):
        self.accelerator.wait_for_everyone()
        selected, source_bootstrap = self._select_checkpoint()
        if source_bootstrap:
            self._verify_source(selected)
        before = selected.stat()
        checkpoint = _load_pt_checkpoint(selected)
        after = selected.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Checkpoint changed while loading")
        keys = BASE_CHECKPOINT_KEYS if source_bootstrap else STAGE_CHECKPOINT_KEYS
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != keys:
            raise RuntimeError("Checkpoint top-level keys do not match source or this new stage")
        model = self.accelerator.unwrap_model(self.model)
        target_online = model.state_dict()
        if source_bootstrap:
            if (checkpoint["update"] != SOURCE_UPDATE or self.optimizer.state_dict()["state"]
                    or self.ema.last_g_stage_update != 0 or self.ema.shadow is not None):
                raise RuntimeError("Use original1300k and a fresh AdamW optimizer")
            online = source_ema_initial_state(checkpoint["ema_model_state_dict"], target_online)
            self._stage_update = self._stage_epoch = self._next_batch_in_epoch = 0
            self._source_checkpoint_size = before.st_size
            self._pending_rng_state = None
        else:
            self._validate_stage_cursor(checkpoint, checkpoint[STAGE_KEY])
            online = checkpoint["model_state_dict"]
            _require_exact_tensor_state(online, target_online, name="stage Online")
            _validate_optimizer_state(checkpoint["optimizer_state_dict"], self.optimizer,
                                      updates=self._stage_update, name="AdamW")
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._set_pending_rng_state(checkpoint)
        model.load_state_dict(online, strict=True)
        if not source_bootstrap:
            self.ema.load_state_dict(checkpoint["ema_model_state_dict"], stage_update=self._stage_update,
                                     tracking=checkpoint[STAGE_KEY]["ema_tracking"])
        self._set_stage_learning_rate()
        for group in self.optimizer.param_groups:
            if (tuple(group["betas"]), group["eps"], group["weight_decay"]) != ((.9, .999), 1e-8, .01):
                raise RuntimeError("Saved AdamW recipe differs")
        del checkpoint
        gc.collect()
        return SOURCE_UPDATE + self._stage_update, source_bootstrap

    def save_checkpoint(self, update, last=False):
        if update != SOURCE_UPDATE + self._stage_update or not 0 <= self._stage_update <= self.total_updates:
            raise RuntimeError("Checkpoint must match the completed update budget")
        if self.ema.last_g_stage_update != self._stage_update:
            raise RuntimeError("EMA must match the completed optimizer update before saving")
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Checkpoint requires a committed optimizer boundary")
        self.accelerator.wait_for_everyone()
        rng_states = self._gather_rng_states()
        error = None
        if self.is_main:
            try:
                payload = {"model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                           "optimizer_state_dict": self.optimizer.state_dict(),
                           "ema_model_state_dict": self.ema.state_dict(),
                           "scheduler_state_dict": self._lr_schedule_state(), "update": update,
                           STAGE_KEY: self._stage_metadata(), RNG_STATE_KEY: rng_states}
                output = Path(self.checkpoint_path)
                output.mkdir(parents=True, exist_ok=True)
                self._atomic_save(payload, output / ("model_last.pt" if last else f"model_{update}.pt"))
            except Exception as exc:
                error = f"Full-model NoGAN save failed; partial evidence preserved: {exc}"
        self._collective_error(error)
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, *, update, loss, loss_dict, applied_lr):
        if self.accelerator.is_local_main_process:
            metrics = {"loss": loss.detach().item(), "lr": applied_lr,
                       "full_msvspec01_nogan/stage_update": self._stage_update,
                       "ema_initialized": int(self.ema.initialized), "ema_age": self.ema.age}
            for key in ("flow_loss", "sf_loss", "ms_v_spec_loss", "ms_v_spec_loss_contribution"):
                metrics[key] = loss_dict[key].detach().item()
            self.accelerator.log(metrics, step=update)
            if self.logger == "tensorboard":
                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, update)

    def train(self, train_dataset, num_workers=8, resumable_with_seed=666):
        self._resumable_seed = resumable_with_seed
        self.accelerator.even_batches = False
        self._original_batch_sampler = DynamicBatchSampler(
            SequentialSampler(train_dataset), self.batch_size_per_gpu, max_samples=self.max_samples,
            random_seed=resumable_with_seed, drop_residual=False)
        loader = DataLoader(train_dataset, collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers > 0,
                            batch_sampler=self._original_batch_sampler)
        self._train_prepared(self.accelerator.prepare(loader))

    def _train_prepared(self, loader):
        self._prepared_batches_per_epoch = len(loader)
        lengths = self.accelerator.gather(torch.tensor([len(loader)], device=self.accelerator.device))
        if lengths.min() <= 0 or lengths.min() != lengths.max():
            raise RuntimeError("All ranks need equal positive prepared-batch counts")
        self.total_updates = self.stage_epochs * self._prepared_batches_per_epoch
        if self.grad_accumulation_steps != 1 or self.accelerator.gradient_accumulation_steps != 1:
            raise RuntimeError("This trial uses one batch per optimizer update")
        _, source_bootstrap = self.load_checkpoint()
        if self._stage_update == self.total_updates:
            self.accelerator.end_training()
            return  # Never forward or rewrite a completed stage.
        if not source_bootstrap and self._next_batch_in_epoch == 0:
            self._restore_pending_rng_state()
        progress = tqdm(total=self.total_updates, initial=self._stage_update, unit="update",
                        disable=not self.accelerator.is_local_main_process)
        while self._stage_update < self.total_updates:
            epoch, skip = self._stage_epoch, self._next_batch_in_epoch
            sampler = getattr(self, "_original_batch_sampler", None)
            if sampler is not None:
                sampler.set_epoch(epoch)
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            self.model.train()
            current = self.accelerator.skip_first_batches(loader, num_batches=skip) if skip else loader
            for index, batch in enumerate(current, start=skip):
                if self._pending_rng_state is not None:
                    self._restore_pending_rng_state()
                applied_lr = self.optimizer.param_groups[0]["lr"]
                with self.accelerator.accumulate(self.model):
                    loss, _, _, metrics = self.model(batch["wav"], text=batch["text"],
                                                     lens=batch["wav_lengths"], noise_scheduler=self.noise_scheduler)
                    # No arithmetic reconstructed from Accelerate-converted BF16
                    # metrics. Objective/gradient correctness is tested offline.
                    if not bool(torch.isfinite(loss).all()):
                        raise RuntimeError("Non-finite training loss; update not committed")
                    self.accelerator.backward(loss)
                    if self.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if getattr(self.optimizer, "step_was_skipped", False):
                        raise RuntimeError("Optimizer update skipped; stage not advanced")
                    self.optimizer.zero_grad(set_to_none=True)
                self._stage_update += 1
                self.ema.after_step(self._stage_update)
                self._stage_epoch = epoch + 1 if index + 1 == len(loader) else epoch
                self._next_batch_in_epoch = 0 if self._stage_epoch != epoch else index + 1
                self._set_stage_learning_rate()
                update = SOURCE_UPDATE + self._stage_update
                self._log_metrics(update=update, loss=loss, loss_dict=metrics, applied_lr=applied_lr)
                progress.update(1)
                final = self._stage_update == self.total_updates
                if final or self._stage_update % self.last_per_updates == 0:
                    self.save_checkpoint(update, last=True)
                if final or self._stage_update % self.save_per_updates == 0:
                    self.save_checkpoint(update)
                if final:
                    break  # Exactly three prepared-loader epochs; no extra update.
        progress.close()
        self.accelerator.end_training()


@hydra.main(version_base="1.3", config_path="../configs", config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    mask_align = _validate_config(cfg)
    vocab_path = REPO_ROOT / "data/Emilia_ZH_EN_pinyin/vocab.txt"
    if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != SOURCE_VOCAB_SHA256:
        raise RuntimeError("Original Emilia pinyin vocabulary identity mismatch")
    source = _resolve_path(cfg.ckpts.source_ckpt, name="source_ckpt")
    output = _resolve_path(cfg.ckpts.save_dir, name="save_dir")
    vocab, count = get_tokenizer(cfg.datasets.name, cfg.model.tokenizer)
    model_cls = hydra.utils.get_class(f"wavegen_v_spec.model.{cfg.model.backbone}")
    model = CFMMSVSpec(transformer=model_cls(**cfg.model.arch, text_num_embeds=count,
                                            wav_frame_len=cfg.model.waveform.wav_frame_len),
                      waveform_kwargs=cfg.model.waveform, vocab_char_map=vocab,
                      **_plain_dict(cfg.model.cfm))
    _validate_model_contract(model, mask_align=mask_align)
    trainer = VSpecTrainer(
        model, source_ckpt=source, resume_checkpoint=cfg.ckpts.resume_ckpt, stage_epochs=cfg.optim.stage_epochs,
        training_contract={"model": _plain_dict(cfg.model), "datasets": _plain_dict(cfg.datasets),
                           "optim": _plain_dict(cfg.optim), "ema": _plain_dict(cfg.ema)},
        learning_rate=cfg.optim.learning_rate, num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates, last_per_updates=cfg.ckpts.last_per_updates,
        keep_last_n_checkpoints=-1, checkpoint_path=str(output), batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type, max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=1, max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger, wandb_project="WaveGen", wandb_run_name=cfg.ckpts.exp_name,
        log_samples=False, bnb_optimizer=False, model_cfg_dict=OmegaConf.to_container(cfg, resolve=True))
    dataset = load_dataset(cfg.datasets.name, cfg.model.tokenizer, waveform_kwargs=cfg.model.waveform)
    trainer.train(dataset, num_workers=cfg.datasets.num_workers, resumable_with_seed=666)


if __name__ == "__main__":
    main()
