"""Generate speech with the fixed four-step WaveGen-Flash release."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


DEFAULT_CHECKPOINT = "ckpts/WaveGen_Flash/model.safetensors"
DEFAULT_CONFIG = "ckpts/WaveGen_Flash/WaveGen_Flash.yaml"
DEFAULT_VOCAB = "ckpts/WaveGen_Flash/vocab.txt"
_HUB_DEFAULTS = {
    "checkpoint_filename": "WaveGen_Flash/model.safetensors",
    "config_filename": "WaveGen_Flash/WaveGen_Flash.yaml",
    "vocab_filename": "WaveGen_Flash/vocab.txt",
}


def add_model_options(parser):
    parser.add_argument("-p", "--checkpoint", "--ckpt-file", "--ckpt_file",
                        help=f"Merged EMA Safetensors; default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("-mc", "--model-config", "--model-cfg", "--model_cfg", "--config",
                        dest="model_config", help=f"Matching model and sampler YAML; default: {DEFAULT_CONFIG}")
    parser.add_argument("-v", "--vocab-file", "--vocab_file", help=f"Matching vocabulary; default: {DEFAULT_VOCAB}")
    parser.add_argument("--repo-id", help="Load the three assets from a Hugging Face repository")
    parser.add_argument("--revision", help="Required with --repo-id; full commit SHA recommended")
    for name, default in _HUB_DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), help=f"Hub asset path; default: {default}")
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true", default=None,
                        help="Require cached Hub assets and a full revision SHA")
    parser.add_argument("--device", help="cuda, cuda:N, or cpu; auto-select CUDA/CPU by default")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compute-final-mel", action=argparse.BooleanOptionalAction, default=None,
                        help="Override final Mel-head computation; the release YAML defaults to false")


def add_sampling_options(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--mel-temperature", type=float,
                        help="Initial Mel noise scale; default comes from the model YAML (0); use 1 for the previous setting")
    parser.add_argument("--max-duration-seconds", type=float,
                        help="Prompt plus generated audio limit; default comes from the model YAML (30 seconds)")
    parser.add_argument("--num-threads", "--num_threads", type=int, default=1)


def validate_options(args):
    if args.repo_id:
        if any(getattr(args, name) is not None for name in ("checkpoint", "model_config", "vocab_file")):
            raise ValueError("--repo-id cannot be combined with local checkpoint, YAML or vocabulary overrides")
        if not args.revision:
            raise ValueError("--revision is required with --repo-id")
    elif any(getattr(args, name) is not None
             for name in ("revision", *_HUB_DEFAULTS, "cache_dir", "local_files_only")):
        raise ValueError("Hub options require --repo-id")
    if isinstance(args.num_threads, bool) or args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    if isinstance(args.seed, bool) or not 0 <= args.seed < 2**63:
        raise ValueError("--seed must be in [0, 2**63)")
    if not math.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("--speed must be finite and positive")
    if args.mel_temperature is not None and (
        isinstance(args.mel_temperature, bool) or not math.isfinite(args.mel_temperature) or args.mel_temperature < 0
    ):
        raise ValueError("--mel-temperature must be finite and nonnegative")
    if args.max_duration_seconds is not None and (
        not math.isfinite(args.max_duration_seconds) or args.max_duration_seconds <= 0
    ):
        raise ValueError("--max-duration-seconds must be finite and positive")
    if args.compute_final_mel is not None and not isinstance(args.compute_final_mel, bool):
        raise ValueError("compute_final_mel must be a boolean")


def load_engine(args, *, engine_class=None):
    validate_options(args)
    if engine_class is None:
        from wavegen_flash.infer.utils_infer import WaveGenFlash

        engine_class = WaveGenFlash
    options = dict(device=args.device, dtype=args.dtype, compute_final_mel=args.compute_final_mel)
    if args.repo_id:
        filenames = {name: getattr(args, name) or default for name, default in _HUB_DEFAULTS.items()}
        return engine_class.from_pretrained(
            args.repo_id, revision=args.revision, cache_dir=args.cache_dir,
            local_files_only=bool(args.local_files_only), **filenames, **options)
    return engine_class.from_checkpoint(
        args.checkpoint or DEFAULT_CHECKPOINT, args.vocab_file or DEFAULT_VOCAB,
        config=args.model_config or DEFAULT_CONFIG, **options)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_model_options(parser)
    add_sampling_options(parser)
    parser.add_argument("-r", "--ref-audio", "--ref_audio", required=True)
    parser.add_argument("-s", "--ref-text", "--ref_text", required=True,
                        help="Exact reference transcript; no ASR is run")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("-t", "--gen-text", "--gen_text", "--text", dest="gen_text")
    target.add_argument("-f", "--gen-file", "--gen_file", dest="gen_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("tests/wavegen_flash.wav"))
    parser.add_argument("--audio-subtype", choices=("PCM_16", "FLOAT"), default="PCM_16")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_options(args)
        output = args.output.expanduser().resolve()
        if output.suffix.lower() != ".wav":
            raise ValueError("--output must have a .wav extension")
        if output.exists() and not args.overwrite:
            raise ValueError(f"Output exists: {output}; use --overwrite explicitly")
        text = args.gen_file.read_text(encoding="utf-8").strip() if args.gen_file else args.gen_text
        if not text or not text.strip():
            raise ValueError("Target text must not be empty")
        import soundfile as sf
        import torch

        torch.set_num_threads(args.num_threads)
        engine = load_engine(args)
        if args.repo_id:
            print(f"Model revision: {engine.hub_repo_id}@{engine.hub_revision}")
        sample_rate, waveform = engine.synthesize(
            args.ref_audio, args.ref_text, text, seed=args.seed, speed=args.speed,
            max_duration_seconds=args.max_duration_seconds, compute_final_mel=args.compute_final_mel,
            mel_temperature=args.mel_temperature)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, waveform, sample_rate, subtype=args.audio_subtype)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Saved {len(waveform) / sample_rate:.2f}s at {sample_rate} Hz with four Euler steps: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
