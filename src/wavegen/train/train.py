# training script.

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from wavegen.model import CFM, Trainer
from wavegen.model.dataset import load_dataset
from wavegen.model.utils import get_tokenizer

@hydra.main(version_base="1.3", config_path="../configs", config_name="WaveGen_SF_Emilia_24k_50Hz_IBDrop02")
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"wavegen.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    cfm_kwargs = getattr(model_cfg.model, "cfm", {}) or {}

    exp_name = model_cfg.ckpts.exp_name
    wandb_resume_id = None

    # set text tokenizer
    tokenizer_path = model_cfg.model.tokenizer_path
    if tokenizer_path is not None:
        vocab_char_map, vocab_size = get_tokenizer(str(Path(tokenizer_path).expanduser()), "custom")
    else:
        vocab_char_map, vocab_size = get_tokenizer(model_cfg.datasets.name, tokenizer)

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, wav_frame_len=model_cfg.model.waveform.wav_frame_len),
        waveform_kwargs=model_cfg.model.waveform,
        vocab_char_map=vocab_char_map,
        **cfm_kwargs,
    )
    
    checkpoint_path = str(Path(model_cfg.ckpts.save_dir).expanduser().resolve())

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="WaveGen",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        accelerate_kwargs={"mixed_precision": model_cfg.optim.mixed_precision},
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    train_dataset = load_dataset(
        str(Path(model_cfg.datasets.path).expanduser().resolve()), tokenizer,
        dataset_type="CustomDatasetPath", waveform_kwargs=model_cfg.model.waveform,
    )
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
