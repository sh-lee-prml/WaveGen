"""Command-line inference: all model and prompt assets are local and explicit."""

from __future__ import annotations

import argparse
from pathlib import Path
from importlib.resources import files
import sys

from wavegen import INFERENCE_METHOD_CHOICES, normalize_inference_method

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def build_parser():
    parser = argparse.ArgumentParser(description="WaveGen 24 kHz waveform inference", allow_abbrev=False)
    parser.add_argument("-c", "--config", help="Inference settings TOML (CLI values override it)",
                        default=str(files("wavegen").joinpath("infer/examples/basic.toml")))
    parser.add_argument("-m", "--model", choices=["WaveGen"], help="Packaged model name")
    parser.add_argument("-mc", "--model_cfg", "--model-cfg", help="Matching model YAML; default: packaged WaveGen_SF_Emilia_24k_50Hz_IBDrop02.yaml")
    parser.add_argument("-p", "--ckpt_file", "--ckpt-file", help="Explicit local full .pt or prefixed EMA .safetensors")
    parser.add_argument("-v", "--vocab_file", "--vocab-file", help="Matching local UTF-8 vocabulary")
    parser.add_argument("-r", "--ref_audio", "--ref-audio", help="Reference speech audio")
    parser.add_argument("-s", "--ref_text", "--ref-text", help="Exact reference transcript; required, no ASR")
    parser.add_argument("-t", "--gen_text", "--gen-text", help="Text to synthesize")
    parser.add_argument("-f", "--gen_file", "--gen-file", help="UTF-8 target text file (takes precedence over gen_text)")
    parser.add_argument("-o", "--output_dir", "--output-dir", help="Output directory; default: tests")
    parser.add_argument("-w", "--output_file", "--output-file", help="WAV filename; default: wavegen.wav")
    parser.add_argument("--nfe_step", "--nfe-step", "--steps", type=int, help="Euler steps; default: 50")
    parser.add_argument("--cfg_strength", "--cfg-strength", type=float, help="Excess CFG; default 3, or 1.25 for scheduled_k3")
    parser.add_argument("--cfg_schedule", "--cfg-schedule", choices=["constant", "scheduled_k3"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--mel_temperature", "--mel-temperature", type=float)
    parser.add_argument("--inference_dropout", "--inference-dropout", type=float)
    parser.add_argument("--inference_method", "--inference-method", choices=INFERENCE_METHOD_CHOICES,
                        help="Prompt conditioning method; default: Prompt_xt_Anchor (Prompt Anchor)")
    parser.add_argument("--device", help="cuda, cuda:N, or cpu; auto CUDA/CPU if omitted")
    parser.add_argument("--dtype", choices=["bf16", "fp32"])
    parser.add_argument("--num_threads", "--num-threads", type=int,
                        help="CPU intra-op threads; default: 1 for native benchmark reproduction")
    parser.add_argument("--checkpoint_state", "--checkpoint-state", choices=["ema", "online"])
    parser.add_argument("--audio_subtype", "--audio-subtype", choices=["PCM_16", "FLOAT"])
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser


def resolve_options(args, parser):
    options = {}
    if args.config:
        with Path(args.config).open("rb") as handle:
            options = tomllib.load(handle)
        known = {action.dest for action in parser._actions}
        unknown = set(options) - known
        if unknown:
            parser.error(f"Unknown TOML settings: {', '.join(sorted(unknown))}")
    options.update({key: value for key, value in vars(args).items() if value is not None})
    defaults = dict(ckpt_file="ckpts/WaveGen/model_1300000.pt",
                    vocab_file="ckpts/WaveGen/vocab.txt", output_dir="tests", output_file="wavegen.wav", nfe_step=50,
                    cfg_schedule="constant", seed=0, speed=1.0, mel_temperature=0.0,
                    inference_dropout=0.2, inference_method="Prompt_xt_Anchor", dtype="bf16", checkpoint_state="ema",
                    audio_subtype="PCM_16", overwrite=False, num_threads=1)
    for key, value in defaults.items():
        options.setdefault(key, value)
    options.setdefault("cfg_strength", 1.25 if options["cfg_schedule"] == "scheduled_k3" else 3.0)
    try:
        options["inference_method"] = normalize_inference_method(options["inference_method"])
    except ValueError as error:
        parser.error(str(error))
    threads = options["num_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        parser.error("--num_threads must be a positive integer")
    for field in ("ref_audio", "vocab_file", "gen_file"):
        value = options.get(field, "")
        if isinstance(value, str) and value.startswith("infer/examples/"):
            options[field] = str(files("wavegen").joinpath(value))
    if options.get("gen_file"):
        options["gen_text"] = Path(options["gen_file"]).read_text(encoding="utf-8").strip()
    for field in ("ckpt_file", "vocab_file", "ref_audio", "ref_text", "gen_text"):
        if not options.get(field):
            parser.error(f"--{field} is required (directly or in TOML)")
    return options


def main(argv=None):
    parser = build_parser()
    options = resolve_options(parser.parse_args(argv), parser)
    output = Path(options["output_dir"]) / options["output_file"]
    if output.suffix.lower() != ".wav":
        parser.error("--output_file must have a .wav extension")
    if output.exists() and not options["overwrite"]:
        parser.error(f"Output exists: {output}; use --overwrite explicitly")
    import soundfile as sf
    import torch
    from wavegen.infer.utils_infer import WaveGen

    torch.set_num_threads(options["num_threads"])
    engine = WaveGen.from_checkpoint(options["ckpt_file"], options["vocab_file"],
        config=options.get("model_cfg"), device=options.get("device"), dtype=options["dtype"],
        use_ema=options["checkpoint_state"] == "ema")
    sr, waveform = engine.synthesize(options["ref_audio"], options["ref_text"], options["gen_text"],
        steps=options["nfe_step"], cfg_strength=options["cfg_strength"],
        cfg_schedule=options["cfg_schedule"], seed=options["seed"], speed=options["speed"],
        mel_temperature=options["mel_temperature"], inference_dropout=options["inference_dropout"],
        inference_method=options["inference_method"])
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), waveform, sr, subtype=options["audio_subtype"])
    print(f"Saved {len(waveform) / sr:.2f}s at {sr} Hz: {output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
