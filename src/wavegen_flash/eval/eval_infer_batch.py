"""Generate the complete supplied metadata list with fixed four-step inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from wavegen_flash.infer.infer_cli import (
    add_model_options, add_sampling_options, load_engine, validate_options,
)
from wavegen_v_spec.eval.common import (
    check_generation, file_identity, metadata_identity, output_lock, read_metadata,
    runtime_settings, source_identity, wav_inventory, write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_model_options(parser)
    add_sampling_options(parser)
    parser.add_argument("--meta", required=True, help="Complete metadata list; no implicit subset")
    parser.add_argument("--dataset", choices=("seedtts", "librispeech"), default="seedtts")
    parser.add_argument("--librispeech-test-clean-path", help="LibriSpeech test-clean audio root")
    parser.add_argument("--output-dir", required=True, help="WAVs and inference manifest are written directly here")
    return parser


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    return {"wavegen_flash": {
        p.relative_to(root).as_posix(): file_identity(p, content=True)["sha256"]
        for p in sorted(root.rglob("*.py"))},
        "wavegen_v_spec": source_identity()}


def run(args, *, engine_class=None, audio_writer=None):
    validate_options(args)
    items = read_metadata(args.meta, dataset=args.dataset,
                          librispeech_test_clean_path=args.librispeech_test_clean_path)
    import torch

    torch.set_num_threads(args.num_threads)
    execution = runtime_settings()
    engine = load_engine(args, engine_class=engine_class)
    if runtime_settings() != execution:
        raise RuntimeError("Thread or TF32 settings changed while loading the inference engine")
    final_mel = (engine.settings["compute_final_mel"] if args.compute_final_mel is None
                 else args.compute_final_mel)
    duration_limit = (engine.settings["max_duration_seconds"] if args.max_duration_seconds is None
                      else args.max_duration_seconds)
    contract = {
        "schema": "wavegen_flash.inference.v1", "model": "WaveGen-Flash",
        "checkpoint": file_identity(engine.checkpoint),
        "config": file_identity(engine.config, content=True),
        "vocab": file_identity(engine.vocab_file, content=True),
        "inputs": metadata_identity(args.meta, items), "source": _source_identity(),
        "dataset": args.dataset,
        "librispeech_test_clean_path": (str(Path(args.librispeech_test_clean_path).resolve())
                                        if args.librispeech_test_clean_path else None),
        "execution": execution, "device": str(engine.device), "dtype": str(engine.dtype),
        "sampler": dict(engine.settings, compute_final_mel=final_mel,
                        max_duration_seconds=duration_limit),
        "seed": args.seed, "speed": args.speed, "wav_subtype": "PCM_16",
        "scope": "all metadata rows",
    }
    if args.repo_id:
        contract["hub"] = {"repo_id": engine.hub_repo_id, "revision": engine.hub_revision}
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        previous = check_generation(output, items)
        if previous.get("contract") != contract:
            raise ValueError(f"Existing output has a different manifest: {output}")
        return {"output": str(output), "count": len(items), "reused": True}
    if audio_writer is None:
        import soundfile as sf

        audio_writer = lambda path, wave, sr: sf.write(path, wave, sr, subtype="PCM_16")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output, ".inference.lock"):
        if any(path.name != ".inference.lock" for path in output.iterdir()):
            raise ValueError(f"Output became nonempty before generation: {output}")
        manifest = {"status": "running", "contract": contract, "ids": [item.utt for item in items]}
        write_json(output / "inference_manifest.json", manifest)
        for index, item in enumerate(items, 1):
            sample_rate, waveform = engine.synthesize(
                str(item.ref_audio), item.ref_text, item.text, seed=args.seed, speed=args.speed,
                max_duration_seconds=duration_limit, compute_final_mel=final_mel)
            audio_writer(output / f"{item.utt}.wav", waveform, sample_rate)
            print(f"WaveGen-Flash: {index}/{len(items)} {item.utt}", flush=True)
        if file_identity(engine.checkpoint) != contract["checkpoint"]:
            raise ValueError("Checkpoint changed during generation")
        if (file_identity(engine.config, content=True) != contract["config"]
                or file_identity(engine.vocab_file, content=True) != contract["vocab"]
                or metadata_identity(args.meta, items) != contract["inputs"]
                or _source_identity() != contract["source"]):
            raise ValueError("Source, model configuration, vocabulary or prompts changed during generation")
        if runtime_settings() != execution:
            raise RuntimeError("Thread or TF32 settings changed during generation")
        manifest.update(status="complete", wavs=wav_inventory(output, items))
        write_json(output / "inference_manifest.json", manifest)
    return {"output": str(output), "count": len(items), "reused": False}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"{'Reused' if result['reused'] else 'Generated'} {result['count']} WAVs: {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
