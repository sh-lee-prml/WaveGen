"""Load the complete Flash EMA Safetensors bundle and synthesize in four steps."""

from contextlib import nullcontext
from importlib.resources import files
import math
from pathlib import Path
import threading
import warnings

import numpy as np
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch.nn import functional as F

from wavegen_v_spec.infer.utils_infer import _local_file, _prepare_prompt, _read_vocabulary, _RNG_LOCK
from wavegen_v_spec.model.backbones.dit import DiT
from wavegen_v_spec.model.utils import convert_char_to_pinyin
from wavegen_flash.model import FlashModel, FlashSampler
from wavegen_flash.model.sampler import validate_mel_temperature


def read_config(path):
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict) or set(config) != {"model", "inference"}:
        raise ValueError("Flash YAML must contain model and inference sections only")
    model, settings = config["model"], config["inference"]
    if not isinstance(model, dict) or not isinstance(settings, dict):
        raise ValueError("model and inference must be YAML mappings")
    expected = dict(num_steps=4, sampling_method="euler", time_shift=3.0,
                    time_grid=[0.0, 0.1, 0.25, 0.5, 1.0], cfg_strength=0.0,
                    inference_bottleneck_dropout=0.0, waveform_temperature=1.0)
    if set(settings) != set(expected) | {"compute_final_mel", "max_duration_seconds", "mel_temperature"}:
        raise ValueError("Unexpected or missing Flash inference settings")
    for key, value in expected.items():
        if isinstance(settings[key], bool) or settings[key] != value:
            raise ValueError(f"WaveGen-Flash requires {key}={value}")
    settings["mel_temperature"] = validate_mel_temperature(settings["mel_temperature"])
    if type(settings['num_steps']) is not int:
        raise ValueError("num_steps must be the integer 4")
    if any(isinstance(t, bool) for t in settings['time_grid']):
        raise ValueError("time_grid entries must be numbers, not booleans")
    if not isinstance(settings["compute_final_mel"], bool):
        raise ValueError("compute_final_mel must be a boolean")
    limit = settings["max_duration_seconds"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("max_duration_seconds must be finite and positive")
    if model.get("name") != "WaveGen_Flash" or model.get("backbone") != "DiT" or model.get("tokenizer") != "pinyin":
        raise ValueError("Expected the WaveGen_Flash DiT/pinyin model config")
    if model.get("cfm") != dict(prediction="x_pred", t_eps=0.02, latents_scale=9.0):
        raise ValueError("Flash requires x_pred, t_eps=0.02 and latents_scale=9.0")
    if model.get("waveform") != dict(target_sample_rate=24000, wav_frame_len=480):
        raise ValueError("Flash requires 24 kHz audio and 480-sample waveform patches")
    return model, settings


class WaveGenFlash:
    """Inference-only EMA model; FP32 weights with optional CUDA BF16 autocast."""

    @classmethod
    def from_pretrained(cls, repo_id, *, revision,
                        checkpoint_filename="WaveGen_Flash/model.safetensors",
                        config_filename="WaveGen_Flash/WaveGen_Flash.yaml",
                        vocab_filename="WaveGen_Flash/vocab.txt", cache_dir=None,
                        local_files_only=False, device=None, dtype="bf16", compute_final_mel=None):
        from wavegen_v_spec.infer.hub import resolve_hub_assets
        assets = resolve_hub_assets(repo_id, revision=revision,
            checkpoint_filename=checkpoint_filename, config_filename=config_filename,
            vocab_filename=vocab_filename, cache_dir=cache_dir, local_files_only=local_files_only)
        instance = cls.from_checkpoint(assets.checkpoint, assets.vocabulary, config=assets.config,
                                      device=device, dtype=dtype, compute_final_mel=compute_final_mel)
        instance.hub_repo_id, instance.hub_revision = assets.repo_id, assets.revision
        return instance

    @classmethod
    def from_checkpoint(cls, checkpoint, vocab_file, config=None, device=None,
                        dtype="bf16", compute_final_mel=None):
        checkpoint = _local_file(checkpoint, "Flash checkpoint")
        if checkpoint.suffix.lower() != ".safetensors":
            raise ValueError("WaveGen-Flash loads the complete EMA .safetensors checkpoint only")
        vocab_file = _local_file(vocab_file, "Vocabulary")
        config = _local_file(config if config is not None else str(files("wavegen_flash").joinpath("configs/WaveGen_Flash.yaml")), "Flash config")
        model_cfg, settings = read_config(config)
        if compute_final_mel is not None:
            if not isinstance(compute_final_mel, bool):
                raise ValueError("compute_final_mel must be a boolean")
            settings["compute_final_mel"] = compute_final_mel
        if dtype not in {"bf16", "fp32"}:
            raise ValueError("dtype must be bf16 or fp32")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu, cuda or cuda:N")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "cpu" and dtype == "bf16":
            warnings.warn("CPU inference uses FP32", RuntimeWarning)
            dtype = "fp32"
        if device.type == "cuda" and dtype == "bf16":
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("This CUDA device does not support BF16; use fp32")
        before = checkpoint.stat()
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("model") != "WaveGen_Flash" or metadata.get("weights") != "ema":
                raise ValueError("Expected the WaveGen_Flash EMA Safetensors release")
        vocabulary = _read_vocabulary(vocab_file)
        arch = dict(model_cfg["arch"], text_num_embeds=len(vocabulary), wav_frame_len=480)
        with _RNG_LOCK, torch.random.fork_rng(devices=[]):
            model = FlashModel(DiT(**arch), vocabulary, t_eps=model_cfg['cfm']['t_eps'],
                               latents_scale=model_cfg['cfm']['latents_scale']).float()
            saved = load_file(str(checkpoint), device="cpu")
            if not saved or any(not key.startswith("ema_model.") or value.dtype != torch.float32
                                for key, value in saved.items()):
                raise ValueError("Flash release must contain only FP32 ema_model tensors")
            state = {key.removeprefix("ema_model."): value for key, value in saved.items()}
            model.load_state_dict(state, strict=True)
        after = checkpoint.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Checkpoint changed while loading")
        instance = cls()
        instance.model = model.requires_grad_(False).to(device).eval()
        instance.device = device
        instance.dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        instance.sample_rate = 24000
        instance.num_steps = 4
        instance.settings = settings
        instance.config, instance.checkpoint, instance.vocab_file = config, checkpoint, vocab_file
        instance.sampler = FlashSampler(instance.model, settings["time_grid"])
        instance._lock = threading.RLock()
        return instance

    def synthesize(self, ref_audio, ref_text, text, *, seed=0, speed=1.0,
                   max_duration_seconds=None, compute_final_mel=None, mel_temperature=None):
        """Return (24000, mono float32 waveform); initial Mel temperature defaults to the YAML."""
        mel_temperature = validate_mel_temperature(
            self.settings["mel_temperature"] if mel_temperature is None else mel_temperature)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if max_duration_seconds is None:
            max_duration_seconds = self.settings["max_duration_seconds"]
        if (isinstance(max_duration_seconds, bool) or not isinstance(max_duration_seconds, (int, float))
                or not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0):
            raise ValueError("max_duration_seconds must be finite and positive")
        if compute_final_mel is None:
            compute_final_mel = self.settings["compute_final_mel"]
        if not isinstance(compute_final_mel, bool):
            raise ValueError("compute_final_mel must be a boolean")
        audio, rms, prompt_length, predicted_length, combined_text = _prepare_prompt(
            ref_audio, ref_text, text, sample_rate=self.sample_rate,
            frame_length=self.model.wav_frame_len, speed=speed)
        frame = self.model.wav_frame_len
        total_length = (predicted_length + frame - 1) // frame * frame
        if total_length > max_duration_seconds * self.sample_rate:
            raise ValueError(f"Prompt plus predicted speech exceeds {max_duration_seconds:g} seconds; split the target text or increase max_duration_seconds")
        wav = F.pad(audio, (0, total_length - prompt_length)).to(self.device)
        mask = torch.arange(total_length, device=self.device)[None, :] >= prompt_length
        lengths = torch.tensor([total_length], device=self.device, dtype=torch.long)
        texts = convert_char_to_pinyin([combined_text])
        cuda_devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                        if self.device.type == "cuda" else [])
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda" and self.dtype == torch.bfloat16 else nullcontext())
        with _RNG_LOCK, self._lock, torch.random.fork_rng(devices=cuda_devices), torch.inference_mode(), autocast:
            torch.random.default_generator.manual_seed(seed)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            self.model.transformer.clear_cache()
            try:
                generated, _ = self.sampler.generate(
                    wav, texts, lengths, mask, compute_final_mel=compute_final_mel,
                    mel_temperature=mel_temperature)
                waveform = generated[0, prompt_length:predicted_length].float().cpu()
                if rms < 0.1:
                    waveform = waveform * rms / 0.1
                result = waveform.numpy().copy()
            finally:
                self.model.transformer.clear_cache()
        if result.size == 0 or not np.isfinite(result).all():
            raise RuntimeError("WaveGen-Flash generated empty or nonfinite audio")
        return self.sample_rate, result
