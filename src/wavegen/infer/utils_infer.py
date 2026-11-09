"""WaveGen inference without downloads, ASR, or external vocoders."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from importlib.resources import files
import inspect
import math
from pathlib import Path
import threading
import warnings

import numpy as np
from omegaconf import OmegaConf
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from wavegen import normalize_inference_method
from wavegen.model import CFM, DiT
from wavegen.model.utils import convert_char_to_pinyin


# torch.random.fork_rng is process-global: serialize even different instances.
_RNG_LOCK = threading.RLock()
_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp32": torch.float32, "float32": torch.float32}


def _local_file(value, label: str) -> Path:
    if not value or str(value).startswith(("http://", "https://", "hf://")):
        raise ValueError(f"{label} must be an explicit local file")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _read_vocabulary(path: Path) -> dict[str, int]:
    tokens = path.read_text(encoding="utf-8").splitlines()
    if not tokens or tokens[0] != " ":
        raise ValueError("Vocabulary must begin with one space token at index 0")
    if len(set(tokens)) != len(tokens) or any(token == "" for token in tokens):
        raise ValueError("Vocabulary contains empty or duplicate tokens")
    return {token: index for index, token in enumerate(tokens)}


def _select_state(checkpoint: Mapping, use_ema: bool) -> dict[str, torch.Tensor]:
    source = "ema_model_state_dict" if use_ema else "model_state_dict"
    raw = checkpoint.get(source)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint has no {source}; another state is never selected silently")
    state = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("Checkpoint state keys must be strings")
        if use_ema and key in {"initted", "step"}:
            if key == "initted" and not bool(value):
                raise ValueError("Checkpoint EMA has not been initialized")
            continue
        if use_ema:
            if not key.startswith("ema_model."):
                raise ValueError(f"EMA key must start with 'ema_model.': {key}")
            key = key.removeprefix("ema_model.")
        if key in state or not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid or duplicate checkpoint tensor: {key}")
        state[key] = value
    if not state:
        raise ValueError("Selected checkpoint state is empty")
    return state


def _load_checkpoint(model: torch.nn.Module, path: Path, use_ema: bool):
    before = path.stat()
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        raw = load_file(str(path), device="cpu")
        # EMA safetensors retain their prefix. Unprefixed files mean online.
        checkpoint = {"ema_model_state_dict" if use_ema else "model_state_dict": raw}
    else:
        load_options = {"map_location": "cpu", "weights_only": True}
        if "mmap" in inspect.signature(torch.load).parameters:
            load_options["mmap"] = True
        checkpoint = torch.load(path, **load_options)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state mapping")
    state = _select_state(checkpoint, use_ema)
    expected = model.state_dict()
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if missing or extra:
        raise ValueError(f"Checkpoint/config mismatch: missing={sorted(missing)[:12]}, "
                         f"unexpected={sorted(extra)[:12]}")
    wrong_shapes = [key for key in expected if expected[key].shape != state[key].shape]
    if wrong_shapes:
        raise ValueError(f"Checkpoint/config tensor shape mismatch: {wrong_shapes[:12]}")
    model.load_state_dict(state, strict=True)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while loading")


def _prepare_prompt(ref_audio, ref_text: str, text: str, *, sample_rate: int,
                    frame_length: int, speed: float, target_rms: float = 0.1):
    """Native UTF-8 length prediction; never inspect target/truth audio."""
    if not isinstance(ref_text, str) or not ref_text.strip():
        raise ValueError("ref_text must contain the supplied reference transcript (ASR is not run)")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    data, sr = sf.read(str(_local_file(ref_audio, "Reference audio")), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy()).mean(dim=0, keepdim=True)
    if audio.numel() == 0 or not torch.isfinite(audio).all():
        raise ValueError("Reference audio must be nonempty and finite")
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms.item() <= 1e-8:
        raise ValueError("Reference audio is silent; provide audible speech")
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    prompt_length = max(audio.shape[-1] // frame_length * frame_length, frame_length)
    if audio.shape[-1] < prompt_length:
        audio = F.pad(audio, (0, prompt_length - audio.shape[-1]))
    else:
        audio = audio[..., :prompt_length]
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text += " "
    total_length = prompt_length + int(
        prompt_length / len(ref_text.encode("utf-8")) * len(text.encode("utf-8")) / speed
    )
    if total_length <= prompt_length:
        raise ValueError("Predicted output is empty; increase text length or lower speed")
    return audio, rms, prompt_length, total_length, ref_text + text


class WaveGen:
    """A loaded WaveGen model. Weights stay FP32; CUDA BF16 uses autocast."""

    def __init__(self, model, *, device, dtype, config, checkpoint, vocab_file, use_ema, tokenizer="pinyin"):
        self.model = model.eval()
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.vocab_file = Path(vocab_file)
        self.use_ema = bool(use_ema)
        self.sample_rate = int(model.target_sample_rate)
        self.tokenizer = tokenizer
        self._lock = threading.RLock()

    @classmethod
    def from_checkpoint(cls, checkpoint, vocab_file, config=None, device=None,
                        dtype="bf16", use_ema=True):
        """Load an explicit local checkpoint, vocabulary and matching YAML.

        Full .pt checkpoints select EMA by default, stripping only the exact
        ``ema_model.`` prefix. No key/shape mismatch or state fallback is allowed.
        CPU inference is explicitly FP32 and is not a CUDA numerical-equivalence claim.
        """
        checkpoint = _local_file(checkpoint, "Checkpoint")
        vocab_file = _local_file(vocab_file, "Vocabulary")
        config = Path(config) if config is not None else Path(str(files("wavegen").joinpath("configs/WaveGen_SF_Emilia_24k_50Hz_IBDrop02.yaml")))
        config = _local_file(config, "Model config")
        requested_dtype = _DTYPES.get(str(dtype).replace("torch.", ""))
        if requested_dtype is None:
            raise ValueError("dtype must be bf16 or fp32")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices are CPU (FP32) and CUDA (FP32/BF16)")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "cpu" and requested_dtype != torch.float32:
            warnings.warn("CPU inference uses FP32, not the requested BF16 CUDA contract", RuntimeWarning)
            requested_dtype = torch.float32
        if device.type == "cuda" and requested_dtype == torch.bfloat16:
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("This CUDA device does not support BF16; request dtype='fp32' explicitly")
        # Resolve only the model subtree: Hydra's training-only `now` resolver
        # is not required (or registered) in this independent inference API.
        model_cfg = OmegaConf.to_container(OmegaConf.load(config).model, resolve=True)
        if model_cfg.get("backbone", "DiT") != "DiT":
            raise ValueError("WaveGen public inference supports the DiT backbone")
        vocabulary = _read_vocabulary(vocab_file)
        arch = dict(model_cfg["arch"])
        waveform = dict(model_cfg["waveform"])
        cfm = dict(model_cfg.get("cfm") or {})
        arch["text_num_embeds"] = len(vocabulary)
        arch["wav_frame_len"] = int(waveform["wav_frame_len"])
        cfm.setdefault("sample_rate", waveform.get("target_sample_rate", 24000))
        with _RNG_LOCK, torch.random.fork_rng(devices=[]):
            model = CFM(transformer=DiT(**arch), waveform_kwargs=waveform,
                        vocab_char_map=vocabulary, **cfm).float()
            _load_checkpoint(model, checkpoint, use_ema)
        model.target_sample_rate = int(waveform.get("target_sample_rate", cfm["sample_rate"]))
        return cls(model.to(device), device=device, dtype=requested_dtype,
                   config=config, checkpoint=checkpoint, vocab_file=vocab_file, use_ema=use_ema,
                   tokenizer=model_cfg.get("tokenizer", "pinyin"))

    def synthesize(self, ref_audio, ref_text, text, steps=50, cfg_strength=3.0,
                   cfg_schedule="constant", seed=0, speed=1.0,
                   mel_temperature=0.0, inference_dropout=0.2, inference_method="Prompt_xt_Anchor",
                   max_duration_frames=4096):
        """Return ``(sample_rate, mono_float32_numpy_waveform)``.

        ``cfg_strength`` is excess guidance: c + g * (c - u). For the registered
        K3 policy pass ``cfg_schedule='scheduled_k3', cfg_strength=1.25``.
        Waveform temperature is one; timestep mapping is power2/shift3.
        ``inference_method`` selects Prompt_xt_Anchor (default) or baseline conditioning.
        ``max_duration_frames`` limits prompt plus generated audio; one frame is 20 ms.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(cfg_strength) or cfg_strength < 0:
            raise ValueError("cfg_strength must be finite and nonnegative")
        if cfg_schedule not in {"constant", "scheduled_k3"}:
            raise ValueError("cfg_schedule must be constant or scheduled_k3")
        inference_method = normalize_inference_method(inference_method)
        if cfg_schedule == "scheduled_k3" and (steps < 3 or cfg_strength < 1e-5):
            raise ValueError("scheduled_k3 requires at least three steps and positive CFG")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not math.isfinite(mel_temperature) or mel_temperature < 0:
            raise ValueError("mel_temperature must be finite and nonnegative")
        if not math.isfinite(inference_dropout) or not 0 <= inference_dropout < 1:
            raise ValueError("inference_dropout must be finite and in [0, 1)")
        if isinstance(max_duration_frames, bool) or not isinstance(max_duration_frames, int) or max_duration_frames < 1:
            raise ValueError("max_duration_frames must be a positive integer")
        audio, rms, prompt_length, total_length, combined_text = _prepare_prompt(
            ref_audio, ref_text, text, sample_rate=self.sample_rate,
            frame_length=self.model.wav_frame_len, speed=speed,
        )
        if total_length > max_duration_frames * self.model.wav_frame_len:
            raise ValueError(f"Predicted duration exceeds {max_duration_frames} frames; increase max_duration_frames or split the target text")
        texts = convert_char_to_pinyin([combined_text]) if self.tokenizer == "pinyin" else [combined_text]
        cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda" and self.dtype == torch.bfloat16 else nullcontext())
        # CFM creates waveform/Mel noise with the global generator, while IB
        # dropout has its own seed-zero generator. Restore caller RNG on exit.
        with _RNG_LOCK, self._lock, torch.random.fork_rng(devices=cuda_devices), torch.inference_mode(), autocast:
            # Seed only generators this call uses; torch.manual_seed would also
            # mutate unrelated CUDA-device RNG states on multi-GPU machines.
            torch.random.default_generator.manual_seed(seed)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            self.model.transformer.clear_cache()
            try:
                generated, trajectory = self.model.sample(
                    cond=audio.to(self.device), text=texts, duration=total_length,
                    steps=steps, cfg_strength=cfg_strength, cfg_schedule=cfg_schedule,
                    inference_method=inference_method,
                    timestep_mapping="power", timestep_power=2.0, shift=3.0,
                    sway_sampling_coef=-1.0, use_epss=False, seed=None,
                    mel_temperature=mel_temperature,
                    inference_bottleneck_dropout=inference_dropout,
                    inference_bottleneck_dropout_seed=0,
                    max_duration=max_duration_frames,
                )
                wave = generated[0, prompt_length:total_length].cpu()
                if rms < 0.1:
                    wave = wave * rms / 0.1
                result = wave.float().numpy().copy()
                del generated, trajectory
            finally:
                self.model.transformer.clear_cache()
        if not np.isfinite(result).all():
            raise RuntimeError("Model generated nonfinite audio")
        return self.sample_rate, result
