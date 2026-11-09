"""Generate a complete test list and optionally evaluate each CFG condition."""

import argparse
import gc
from importlib.resources import files
import json
import math
from pathlib import Path
import re

from wavegen_v_spec import INFERENCE_METHOD_CHOICES, normalize_inference_method

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .common import (check_generation, file_identity, metadata_identity,
                     output_lock, read_metadata, runtime_settings, source_identity, wav_inventory, write_json)


_CONFIG_FIELDS = {
    "inference": {
        "checkpoint", "vocab_file", "config", "meta", "output_dir", "conditions", "steps",
        "cfg_strength", "scheduled_cfg_strength", "seed", "speed", "mel_temperature",
        "inference_dropout", "inference_method", "dtype", "device", "num_threads", "no_ema",
        "dataset", "librispeech_test_clean_path", "max_duration_frames", "compute_final_mel",
    },
    "evaluation": {
        "mode", "eval_task", "lang", "eval_device", "asr_model", "asr_compute_type",
        "s3prl_repo", "wavlm_checkpoint", "sim_checkpoint", "utmos_repo", "utmos_checkpoint",
    },
}


class _BatchArgumentParser(argparse.ArgumentParser):
    """Load TOML defaults before parsing CLI overrides, including explicit zero/False."""

    def parse_args(self, args=None, namespace=None):
        probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        probe.add_argument("--eval-config")
        selected, _ = probe.parse_known_args(args)
        if not selected.eval_config:
            return super().parse_args(args, namespace)
        try:
            with Path(selected.eval_config).open("rb") as handle:
                document = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            self.error(f"Cannot read --eval-config: {error}")
        unknown = set(document) - set(_CONFIG_FIELDS)
        if unknown:
            self.error(f"Unknown TOML sections: {', '.join(sorted(unknown))}")
        actions = {action.dest: action for action in self._actions}
        configured = {}
        for section, values in document.items():
            if not isinstance(values, dict):
                self.error(f"TOML [{section}] must be a table")
            unknown = set(values) - _CONFIG_FIELDS[section]
            if unknown:
                self.error(f"Unknown TOML [{section}] settings: {', '.join(sorted(unknown))}")
            for key, value in values.items():
                action = actions[key]
                if key in ("no_ema", "compute_final_mel"):
                    valid = type(value) is bool
                elif action.type is int:
                    valid = type(value) is int
                elif action.type is float:
                    valid = type(value) in (int, float) and math.isfinite(value)
                else:
                    valid = isinstance(value, str)
                if not valid:
                    self.error(f"Invalid TOML type or nonfinite value for [{section}].{key}")
                choices = ("infer-only", "full-eval") if key == "mode" else action.choices
                if choices is not None and value not in choices:
                    self.error(f"Invalid TOML [{section}].{key}: choose from {', '.join(choices)}")
                if key in ("meta", "output_dir") and not value.strip():
                    self.error(f"TOML [{section}].{key} must not be empty")
                configured[key] = action.type(value) if action.type is not None else value
        original_defaults = self._defaults.copy()
        original_actions = [(action, action.default, action.required) for action in self._actions]
        try:
            self.set_defaults(**configured)
            for action in self._actions:
                if action.dest in configured:
                    action.required = False
            return super().parse_args(args, namespace)
        finally:
            self._defaults = original_defaults
            for action, default, required in original_actions:
                action.default, action.required = default, required


def build_parser():
    parser = _BatchArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--eval-config", help="TOML [inference]/[evaluation] settings; CLI overrides")
    parser.add_argument("--checkpoint", default="ckpts/WaveGen_VSpec/model_1400000.safetensors")
    parser.add_argument("--vocab-file", default="ckpts/WaveGen_VSpec/vocab.txt")
    parser.add_argument("--config", help="Matching model YAML (separate from --eval-config)")
    parser.add_argument("--meta", required=True, help="Complete metadata list; no implicit subset")
    parser.add_argument("--dataset", choices=("seedtts", "librispeech"), default="seedtts")
    parser.add_argument("--librispeech-test-clean-path", help="LibriSpeech test-clean audio root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", choices=("both", "constant", "scheduled_k3"), default="scheduled_k3")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-strength", type=float, default=3.0, help="Constant excess CFG")
    parser.add_argument("--scheduled-cfg-strength", type=float, default=1.25, help="K3 base excess CFG")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-duration-frames", type=int, default=4096,
                        help="Prompt plus output length limit in 20 ms frames; use 5120 for Seed-TTS ZH hard")
    parser.add_argument("--mel-temperature", type=float, default=0.0)
    parser.add_argument("--compute-final-mel", action=argparse.BooleanOptionalAction, default=False,
                        help="Compute the unused final-step Mel head; skipped by default")
    parser.add_argument("--inference-dropout", type=float, default=0.2)
    parser.add_argument("--inference-method", choices=INFERENCE_METHOD_CHOICES, default="Prompt_xt_Anchor",
                        help="Prompt conditioning method; default: Prompt_xt_Anchor (Prompt Anchor)")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-threads", type=int, default=1,
                        help="PyTorch CPU threads, including prompt RMS preprocessing (default: 1)")
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--no-ema", dest="no_ema", action="store_true", help="Use online checkpoint weights")
    state.add_argument("--ema", dest="no_ema", action="store_false", help="Use EMA checkpoint weights (default)")
    parser.set_defaults(no_ema=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--infer-only", dest="mode", action="store_const", const="infer-only",
                      help="Generate audio only (default)")
    mode.add_argument("--full-eval", dest="mode", action="store_const", const="full-eval",
                      help="Generate audio, then score each selected CFG condition")
    parser.set_defaults(mode="infer-only")
    parser.add_argument("--eval-task", choices=("all", "wer", "sim", "utmos"), default="all")
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--eval-device", default="cpu", help="Metric device: cpu or cuda[:index]")
    parser.add_argument("--asr-model", help="Local Faster-Whisper large-v3 or Paraformer-zh directory")
    parser.add_argument("--asr-compute-type", help="Faster-Whisper compute type; default float16 CUDA / int8 CPU")
    parser.add_argument("--s3prl-repo", help="Local S3PRL checkout")
    parser.add_argument("--wavlm-checkpoint", help="Local upstream WavLM Large weights")
    parser.add_argument("--sim-checkpoint", "--wavlm-ckpt-dir", "--wavlm_ckpt_dir", dest="sim_checkpoint",
                        help="Local ECAPA/WavLM speaker fine-tuned checkpoint")
    parser.add_argument("--utmos-repo", help="Local SpeechMOS v1.2.0 checkout")
    parser.add_argument("--utmos-checkpoint", help="Local utmos22_strong_step7459_v1.pt")
    return parser


def run(args, *, engine_factory=None, audio_writer=None):
    args.inference_method = normalize_inference_method(args.inference_method)
    args.compute_final_mel = getattr(args, "compute_final_mel", False)
    if not isinstance(args.compute_final_mel, bool):
        raise ValueError("compute_final_mel must be a boolean")
    args.max_duration_frames = getattr(args, "max_duration_frames", 4096)
    if isinstance(args.max_duration_frames, bool) or not isinstance(args.max_duration_frames, int) or args.max_duration_frames < 1:
        raise ValueError("max_duration_frames must be a positive integer")
    if args.num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    if args.steps < 1 or not 0 <= args.seed < 2**63:
        raise ValueError("steps must be positive and seed must be in [0, 2**63)")
    if not math.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("speed must be finite and positive")
    for name in ("cfg_strength", "scheduled_cfg_strength", "mel_temperature", "inference_dropout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.inference_dropout >= 1:
        raise ValueError("inference_dropout must be less than one")
    if args.conditions != "constant" and (args.steps < 3 or args.scheduled_cfg_strength < 1e-5):
        raise ValueError("scheduled_k3 requires at least three steps and positive CFG")
    config = Path(args.config) if args.config else Path(str(files("wavegen_v_spec").joinpath("configs/WaveGen_VSpec.yaml")))
    dataset = getattr(args, "dataset", "seedtts")
    librispeech_root = getattr(args, "librispeech_test_clean_path", None)
    items = read_metadata(args.meta, dataset=dataset, librispeech_test_clean_path=librispeech_root)
    import torch

    # Reduction order can change prompt RMS and the final BF16 waveform.
    torch.set_num_threads(args.num_threads)
    execution = runtime_settings()
    common = {"schema": "wavegen_v_spec.seedtts.inference.v1", "checkpoint": file_identity(args.checkpoint),
              "vocab": file_identity(args.vocab_file, content=True), "config": file_identity(config, content=True),
              "inputs": metadata_identity(args.meta, items), "source": source_identity(),
              "dataset": dataset,
              "librispeech_test_clean_path": str(Path(librispeech_root).resolve()) if librispeech_root else None,
              "execution": execution,
              "dtype": args.dtype, "device": args.device, "use_ema": not args.no_ema,
              "steps": args.steps, "seed": args.seed, "speed": args.speed,
              "max_duration_frames": args.max_duration_frames,
              "mel_temperature": args.mel_temperature, "inference_dropout": args.inference_dropout,
              "inference_method": args.inference_method, "compute_final_mel": args.compute_final_mel,
              "waveform_temperature": 1.0, "timestep_mapping": "power", "timestep_power": 2.0,
              "shift": 3.0, "wav_subtype": "PCM_16", "scope": "all metadata rows"}
    conditions = ("constant", "scheduled_k3") if args.conditions == "both" else (args.conditions,)
    plans, results = [], {}
    for condition in conditions:
        strength = args.cfg_strength if condition == "constant" else args.scheduled_cfg_strength
        contract = dict(common, cfg_schedule=condition, cfg_strength=strength)
        output = Path(args.output_dir).resolve() / condition
        if output.exists() and any(output.iterdir()):
            previous = check_generation(output, items)
            if previous.get("contract") != contract:
                raise ValueError(f"Existing output has a different manifest: {output}")
            results[condition] = {"output": str(output), "count": len(items), "reused": True}
        else:
            plans.append((condition, strength, contract, output))
    if not plans:
        return results
    if engine_factory is None:
        from wavegen_v_spec.infer.utils_infer import WaveGen
        engine_factory = WaveGen.from_checkpoint
    if audio_writer is None:
        import soundfile as sf
        audio_writer = lambda path, wave, sr: sf.write(path, wave, sr, subtype="PCM_16")
    engine = engine_factory(args.checkpoint, args.vocab_file, config=str(config), device=args.device,
                            dtype=args.dtype, use_ema=not args.no_ema)
    if runtime_settings() != execution:
        raise RuntimeError("Thread or TF32 settings changed while loading the inference engine")
    for condition, strength, contract, output in plans:
        output.mkdir(parents=True, exist_ok=True)
        with output_lock(output, ".inference.lock"):
            if any(p.name != ".inference.lock" for p in output.iterdir()):
                raise ValueError(f"Output became nonempty before generation: {output}")
            manifest = {"status": "running", "contract": contract, "ids": [x.utt for x in items],
                        "runtime": {"device": str(getattr(engine, "device", args.device)),
                                    "dtype": str(getattr(engine, "dtype", args.dtype)), **execution}}
            write_json(output / "inference_manifest.json", manifest)
            for index, item in enumerate(items, 1):
                sample_rate, wave = engine.synthesize(
                    ref_audio=str(item.ref_audio), ref_text=item.ref_text, text=item.text,
                    steps=args.steps, cfg_strength=strength, cfg_schedule=condition, seed=args.seed,
                    speed=args.speed, mel_temperature=args.mel_temperature,
                    inference_dropout=args.inference_dropout, inference_method=args.inference_method,
                    max_duration_frames=args.max_duration_frames, compute_final_mel=args.compute_final_mel)
                audio_writer(output / f"{item.utt}.wav", wave, sample_rate)
                print(f"{condition}: {index}/{len(items)} {item.utt}", flush=True)
            if file_identity(args.checkpoint) != common["checkpoint"]:
                raise ValueError("Checkpoint changed during generation")
            if runtime_settings() != execution:
                raise RuntimeError("Thread or TF32 settings changed during generation")
            if (file_identity(args.vocab_file, content=True) != common["vocab"]
                    or file_identity(config, content=True) != common["config"]
                    or metadata_identity(args.meta, items) != common["inputs"]):
                raise ValueError("Config, vocabulary or prompt assets changed during generation")
            manifest.update(status="complete", wavs=wav_inventory(output, items))
            write_json(output / "inference_manifest.json", manifest)
        results[condition] = {"output": str(output), "count": len(items), "reused": False}
    return results


def _metric_args(args, output):
    return argparse.Namespace(
        meta=args.meta, gen_wav_dir=output, dataset=getattr(args, "dataset", "seedtts"),
        librispeech_test_clean_path=getattr(args, "librispeech_test_clean_path", None),
        eval_task=args.eval_task, lang=args.lang, device=args.eval_device, num_threads=args.num_threads,
        asr_model=args.asr_model, asr_compute_type=args.asr_compute_type, s3prl_repo=args.s3prl_repo,
        wavlm_checkpoint=args.wavlm_checkpoint, sim_checkpoint=args.sim_checkpoint,
        utmos_repo=args.utmos_repo, utmos_checkpoint=args.utmos_checkpoint,
    )


def _release_generation_memory():
    # run() owns the generator; its local engine is gone before this collection.
    import torch

    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def run_pipeline(args, *, engine_factory=None, audio_writer=None):
    mode = getattr(args, "mode", "infer-only")
    if mode not in ("infer-only", "full-eval"):
        raise ValueError("mode must be infer-only or full-eval")
    if getattr(args, "dataset", "seedtts") == "librispeech" and getattr(args, "lang", "en") != "en":
        raise ValueError("LibriSpeech evaluation requires --lang en")
    if mode == "infer-only":
        return run(args, engine_factory=engine_factory, audio_writer=audio_writer)

    from . import eval_seedtts_testset
    import torch

    if args.num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.eval_device):
        raise ValueError("eval_device must be cpu or cuda[:index]")
    if args.eval_task not in ("all", "wer", "sim", "utmos") or args.lang not in ("en", "zh"):
        raise ValueError("Invalid evaluation task or language")
    torch.set_num_threads(args.num_threads)
    selected = ("wer", "sim", "utmos") if args.eval_task == "all" else (args.eval_task,)
    options = _metric_args(args, args.output_dir)
    # Check every selected model path before spending time on generation.
    for metric in selected:
        identity = eval_seedtts_testset.scorer_identity(metric, options)
        if metric == "wer" and not identity["asr_model"]["files"]:
            raise ValueError("--asr-model must contain local model files")

    generation = run(args, engine_factory=engine_factory, audio_writer=audio_writer)
    _release_generation_memory()
    evaluation = {}
    for condition, result in generation.items():
        evaluation[condition] = eval_seedtts_testset.run(_metric_args(args, result["output"]))
    return {"generation": generation, "evaluation": evaluation}


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(json.dumps(run_pipeline(args), indent=2))


if __name__ == "__main__":
    main()
