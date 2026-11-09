"""Checkpoint, optimizer and rank-RNG utilities for WaveGen fine-tuning.

Serialized source identities are kept for compatibility with the released model.
The original 1300k checkpoint is used only to initialize a fresh training stage.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

# CLI commands run from the project root; installed wheels keep paths portable.
REPO_ROOT = Path.cwd()



SOURCE_UPDATE = 1_300_000

# New checkpoints use the public package name. The old tag is accepted only
# while reading existing full-training checkpoints.
STAGE_FORMAT = "wavegen_v_spec"
LEGACY_STAGE_FORMAT = "wavtts_sf_full_msvspec01_nogan_ema2000_3ep_v1_from1300k"


SOURCE_CHECKPOINT_SIZE = 10_890_285_502


SOURCE_CHECKPOINT_SHA256 = "9aec19b1826e77abeb6e7f0314d40c48103663de5046b30d2d5182e6a0117f44"


SOURCE_VOCAB_SHA256 = "836aabf4c2306d215ce4331524d216e29ccecbc9588c75c73d9b3e77439c2692"


BASE_CHECKPOINT_KEYS = frozenset({
    "model_state_dict", "optimizer_state_dict", "ema_model_state_dict",
    "scheduler_state_dict", "update",
})


SPECTRAL_CONTRACT = {
    "implementation": "wavegen.model.modules.MultiScaleLinearSpectrogramLoss",
    "name": "legacy_WaveGen_MS-V-Spec",
    "space": "x_pred_error",
    "amplitude": "divide_prediction_and_target_by_latents_scale",
    "window_lengths": [32, 64, 128, 256, 512, 1024, 2048],
    "hop_lengths": [8, 16, 32, 64, 128, 256, 512],
    "band_counts": [5, 10, 20, 40, 80, 160, 320],
    "weight": 0.05,
    "aux_mel_enabled": False,
    "mask_alignment": 512,
    "gan_enabled": False,
}


def _resolve_path(value: Any, *, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        converted = OmegaConf.to_container(value, resolve=True)
    elif isinstance(value, Mapping):
        converted = dict(value)
    else:
        raise TypeError(f"Expected a mapping, got {type(value).__name__}")
    if not isinstance(converted, dict):
        raise TypeError(f"Expected a mapping, got {type(converted).__name__}")
    return converted


def _load_pt_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:  # pragma: no cover - older Torch without mmap
        return torch.load(path, map_location="cpu", weights_only=True)


def _validate_optimizer_state(state, optimizer, *, updates, name):
    """Reject silently empty/stale AdamW moments on a nonzero stage resume."""
    groups = state.get("param_groups", [])
    if len(groups) != len(optimizer.param_groups):
        raise RuntimeError(f"{name}: optimizer group count mismatch")
    parameters = {}
    for saved, live in zip(groups, optimizer.param_groups):
        ids = saved.get("params", [])
        if len(ids) != len(live["params"]):
            raise RuntimeError(f"{name}: optimizer parameter count mismatch")
        for index, parameter in zip(ids, live["params"]):
            if index in parameters:
                raise RuntimeError(f"{name}: duplicate optimizer parameter ID")
            parameters[index] = parameter
    moments = state.get("state", {})
    if (updates == 0 and moments) or (updates > 0 and not moments):
        raise RuntimeError(f"{name}: missing or unexpected AdamW moments at update {updates}")
    steps = []
    for index, values in moments.items():
        if index not in parameters or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"{name}: incomplete or foreign AdamW state")
        step = float(torch.as_tensor(values["step"]).item())
        if not math.isfinite(step) or not step.is_integer() or not 1 <= step <= updates:
            raise RuntimeError(f"{name}: invalid AdamW step {step}")
        steps.append(int(step))
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = values[key]
            if not torch.is_tensor(tensor) or tensor.shape != parameters[index].shape:
                raise RuntimeError(f"{name}: AdamW moment shape mismatch")
    if updates > 0 and max(steps) != updates:
        raise RuntimeError(f"{name}: stale AdamW moments")


def _strip_source_aux_mel_buffers(state_dict: Mapping[str, Any], *, ema: bool) -> dict[str, Any]:
    """Remove exactly14 parameter-free legacy buffers, only on source bootstrap."""
    prefix = "ema_model.aux_mel_loss." if ema else "aux_mel_loss."
    shapes = {}
    for index, (window, bands) in enumerate(zip(
            SPECTRAL_CONTRACT["window_lengths"], SPECTRAL_CONTRACT["band_counts"])):
        root = f"{prefix}mel_transforms.{index}"
        shapes[f"{root}.spectrogram.window"] = (window,)
        shapes[f"{root}.mel_scale.fb"] = (window // 2 + 1, bands)
    found = {key for key in state_dict if key.startswith(prefix)}
    if found != set(shapes):
        raise RuntimeError("Source auxiliary-Mel buffer keys differ from the exact14-buffer contract")
    for key, shape in shapes.items():
        value = state_dict[key]
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != torch.float32:
            raise RuntimeError(f"Source auxiliary-Mel buffer shape/dtype mismatch: {key}")
    # mel_f.* conditioning buffers and every learned tensor are retained verbatim.
    return {key: value for key, value in state_dict.items() if key not in shapes}


def relocated_stage_source(saved_contract, expected_contract, source_size):
    """Accept relocation only for the exact registered source and unchanged recipe.

    A resumed stage contains its own full Online/EMA/optimizer states, so the
    original source checkpoint need not be reopened. Its original path remains
    in subsequently saved provenance. Only the location may differ; the source
    identity, every training setting and all stage-state checks stay strict.
    The legacy format tag is equivalent to the public tag when loading; this
    normalization never changes the input checkpoint or its tensor states.
    """
    if not isinstance(saved_contract, Mapping) or not isinstance(expected_contract, Mapping):
        raise RuntimeError("Stage contract must be a mapping")
    if (type(source_size) is not int or source_size != SOURCE_CHECKPOINT_SIZE
            or saved_contract.get("source_sha256") != SOURCE_CHECKPOINT_SHA256
            or expected_contract.get("source_sha256") != SOURCE_CHECKPOINT_SHA256):
        raise RuntimeError("Relocated source requires the registered 1300k SHA256 and size")
    original_path = saved_contract.get("source_checkpoint")
    current_path = expected_contract.get("source_checkpoint")
    if (not isinstance(original_path, str) or not original_path.strip()
            or not isinstance(current_path, str) or not current_path.strip()):
        raise RuntimeError("Stage source checkpoint path must be nonempty")
    original = dict(saved_contract)
    current = dict(expected_contract)
    original.pop("source_checkpoint")
    current.pop("source_checkpoint")
    if (original.get("format") == LEGACY_STAGE_FORMAT
            and current.get("format") == STAGE_FORMAT):
        original["format"] = STAGE_FORMAT
    if original != current:
        raise RuntimeError("Stage contract differs beyond the relocated source checkpoint path")
    return original_path


class CheckpointIO:
    """Shared atomic checkpoint and complete per-rank RNG operations."""

    @property
    def is_main(self):
        return self.accelerator.is_main_process


    def _select_checkpoint(self) -> tuple[Path, bool]:
        output_dir = Path(self.checkpoint_path).resolve()
        model_last = output_dir / "model_last.pt"
        incomplete_writes = (
            sorted(output_dir.glob(".model*.pt.*.tmp"))
            if output_dir.is_dir()
            else []
        )
        if incomplete_writes:
            raise RuntimeError(
                "MSVSpec output contains an incomplete atomic checkpoint write: "
                f"{[path.name for path in incomplete_writes]}. Review/remove it "
                "explicitly instead of silently resuming an older checkpoint."
            )
        explicit = getattr(self, "resume_checkpoint", None)
        if explicit is not None:
            explicit = Path(explicit).resolve()
            if not explicit.is_file() or explicit.suffix.lower() != ".pt":
                raise FileNotFoundError(f"Full fine-tuning checkpoint not found: {explicit}")
            return explicit, False
        if model_last.is_file():
            return model_last, False
        if output_dir.is_dir() and any(output_dir.glob("*.pt")):
            raise RuntimeError(
                f"{output_dir} contains checkpoints but no model_last.pt; restore "
                "the stage model_last.pt rather than silently restarting the stage"
            )
        if not self.source_ckpt.is_file():
            raise FileNotFoundError(f"Full source checkpoint not found: {self.source_ckpt}")
        return self.source_ckpt, True


    def _capture_local_rng_state(self) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        device_rng = None
        if self.accelerator.device.type == "cuda":
            device_rng = torch.cuda.get_rng_state(self.accelerator.device).cpu()
        return {
            "rank": int(self.accelerator.process_index),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_device": device_rng,
            "python": random.getstate(),
            "numpy": {
                "bit_generator": str(numpy_state[0]),
                "keys": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
        }


    def _gather_rng_states(self) -> list[dict[str, Any]] | None:
        local_state = self._capture_local_rng_state()
        if dist.is_available() and dist.is_initialized():
            gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_state)
            if self.is_main:
                states = [state for state in gathered if state is not None]
                states.sort(key=lambda state: int(state["rank"]))
                return states
            return None
        return [local_state] if self.is_main else None


    def _restore_pending_rng_state(self) -> None:
        if self._pending_rng_state is None:
            return
        state = self._pending_rng_state
        torch_cpu = state.get("torch_cpu")
        if not isinstance(torch_cpu, torch.Tensor):
            raise RuntimeError("Stage RNG state is missing torch_cpu")
        torch.set_rng_state(torch_cpu.cpu())

        torch_device = state.get("torch_device")
        if self.accelerator.device.type == "cuda":
            if not isinstance(torch_device, torch.Tensor):
                raise RuntimeError("Stage RNG state is missing torch_device for CUDA resume")
            torch.cuda.set_rng_state(torch_device.cpu(), self.accelerator.device)

        python_state = state.get("python")
        if not isinstance(python_state, tuple):
            raise RuntimeError("Stage RNG state is missing Python random state")
        random.setstate(python_state)

        numpy_state = state.get("numpy")
        if not isinstance(numpy_state, Mapping):
            raise RuntimeError("Stage RNG state is missing NumPy state")
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        self._pending_rng_state = None


    def _collective_error(self, error):
        errors = [error]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            raise RuntimeError(errors[0])


    def _verify_source(self, path):
        error = None
        if self.is_main:
            try:
                before = path.stat()
                if before.st_size != SOURCE_CHECKPOINT_SIZE:
                    raise RuntimeError("1300k source checkpoint size mismatch")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                        digest.update(chunk)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError("1300k checkpoint changed during source verification")
                if digest.hexdigest() != SOURCE_CHECKPOINT_SHA256:
                    raise RuntimeError("1300k source checkpoint SHA256 mismatch")
            except Exception as exc:
                error = f"Source verification failed: {exc}"
        self._collective_error(error)


    def _atomic_save(self, checkpoint, destination):
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        # Exclusive create + fsync. Keep failed temporary evidence for review.
        with temporary.open("xb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
