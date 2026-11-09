"""Score complete generated Seed-TTS or LibriSpeech audio using local evaluation assets."""

import argparse
import json
from pathlib import Path
import re

from .common import (aggregate, check_generation, file_identity, load_json,
                     metadata_identity, output_lock, read_metadata, runtime_settings, source_identity, write_json)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--dataset", choices=("seedtts", "librispeech"), default="seedtts")
    parser.add_argument("--librispeech-test-clean-path", help="LibriSpeech test-clean audio root")
    parser.add_argument("-g", "--gen-wav-dir", "--gen_wav_dir", required=True)
    parser.add_argument("-e", "--eval-task", "--eval_task", choices=("all", "wer", "sim", "utmos"), default="all")
    parser.add_argument("-l", "--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument("--num-threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    parser.add_argument("--asr-model", help="Local Faster-Whisper large-v3 or Paraformer-zh directory")
    parser.add_argument("--asr-compute-type", help="Faster-Whisper compute type; defaults to float16 CUDA / int8 CPU")
    parser.add_argument("--s3prl-repo", help="Trusted local S3PRL checkout containing hubconf.py")
    parser.add_argument("--wavlm-checkpoint", help="Upstream WavLM Large checkpoint; separate from the SIM head")
    parser.add_argument("--sim-checkpoint", help="ECAPA/WavLM fine-tuned speaker checkpoint")
    parser.add_argument("--utmos-repo", help="Trusted local SpeechMOS v1.2.0 checkout")
    parser.add_argument("--utmos-checkpoint", help="Local utmos22_strong_step7459_v1.pt")
    return parser


def directory_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Expected a model directory: {path}")
    return {"path": str(path), "files": {
        p.relative_to(path).as_posix(): file_identity(p) for p in sorted(path.rglob("*")) if p.is_file()}}


def scorer_identity(metric, args):
    required = {"wer": ("asr_model",),
                "sim": ("s3prl_repo", "wavlm_checkpoint", "sim_checkpoint"),
                "utmos": ("utmos_repo", "utmos_checkpoint")}[metric]
    missing = ["--" + key.replace("_", "-") for key in required if not getattr(args, key, None)]
    if missing:
        raise ValueError(f"{metric.upper()} requires {' '.join(missing)}")
    identity = {"metric": metric, "language": args.lang, "device": args.device,
                "execution": runtime_settings()}
    for key in required:
        path = Path(getattr(args, key))
        if key == "s3prl_repo":
            identity[key] = file_identity(path / "s3prl/upstream/wavlm/hubconf.py", content=True)
        elif key.endswith("_repo"):
            identity[key] = file_identity(path / "hubconf.py", content=True)
        elif key == "asr_model":
            identity[key] = directory_identity(path)
        else:
            identity[key] = file_identity(path)
    if metric == "wer":
        identity["compute_type"] = (args.asr_compute_type or ("float16" if args.device.startswith("cuda") else "int8")) if args.lang == "en" else "funasr-default"
        identity["aggregation"] = "macro mean utterance WER fraction; report percent, uncapped"
        identity["normalization"] = "Seed-TTS punctuation removal; English lowercase / Chinese characters"
    elif metric == "sim":
        identity["reference"] = "prompt audio, not target audio"
        identity["decoding"] = "soundfile float32 channels-first tensor; mean channels"
        identity["resampling"] = "torchaudio Resample to 16000 Hz"
    else:
        identity["decoding"] = "librosa native sample rate mono; predictor resamples to 16000 Hz"
    return identity


def run(args, *, scorers=None):
    if args.num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
        raise ValueError("device must be cpu or cuda[:index]")
    dataset = getattr(args, "dataset", "seedtts")
    if dataset == "librispeech" and args.lang != "en":
        raise ValueError("LibriSpeech evaluation requires --lang en")
    items = read_metadata(args.meta, dataset=dataset,
                          librispeech_test_clean_path=getattr(args, "librispeech_test_clean_path", None))
    directory = Path(args.gen_wav_dir).resolve(strict=True)
    generation = check_generation(directory, items)
    if generation["contract"].get("dataset", "seedtts") != dataset:
        raise ValueError("Evaluation dataset differs from generation")
    inputs = metadata_identity(args.meta, items)
    if generation["contract"]["inputs"] != inputs:
        raise ValueError("Evaluation metadata/reference assets differ from generation")
    import torch

    torch.set_num_threads(args.num_threads)
    metrics = ("wer", "sim", "utmos") if args.eval_task == "all" else (args.eval_task,)
    identities = {metric: scorer_identity(metric, args) for metric in metrics}
    if scorers is None:
        from .utils_eval import run_asr_wer, run_sim, run_utmos
        scorers = {
            "wer": lambda: run_asr_wer(items, directory, lang=args.lang, asr_model=args.asr_model,
                                       device=args.device, compute_type=args.asr_compute_type),
            "sim": lambda: run_sim(items, directory, s3prl_repo=args.s3prl_repo,
                                   wavlm_checkpoint=args.wavlm_checkpoint,
                                   sim_checkpoint=args.sim_checkpoint, device=args.device),
            "utmos": lambda: run_utmos(items, directory, utmos_repo=args.utmos_repo,
                                       utmos_checkpoint=args.utmos_checkpoint, device=args.device),
        }
    summaries = {}
    for metric in metrics:
        manifest_path = directory / f"_{metric}_manifest.json"
        result_path = directory / f"_{metric}_results.jsonl"
        contract = {"schema": "wavegen_v_spec.seedtts.metrics.v1", "scorer": identities[metric],
                    "generation": file_identity(directory / "inference_manifest.json", content=True),
                    "source": source_identity()}
        with output_lock(directory, ".metrics.lock"):
            reuse = False
            if manifest_path.exists() or result_path.exists():
                if not manifest_path.exists():
                    raise ValueError(f"Unregistered {metric} results exist; preserve them for inspection")
                previous = load_json(manifest_path)
                if previous.get("contract") != contract:
                    raise ValueError(f"Existing {metric} manifest does not match this evaluation")
                if previous.get("status") == "running" and not result_path.exists():
                    attempt = 1
                    while (directory / f"_{metric}_interrupted_{attempt:03d}_manifest.json").exists():
                        attempt += 1
                    archive = directory / f"_{metric}_interrupted_{attempt:03d}_manifest.json"
                    with archive.open("xb") as handle:
                        handle.write(manifest_path.read_bytes())
                elif previous.get("status") == "complete" and result_path.exists():
                    reuse = True
                else:
                    raise ValueError(f"Partial {metric} results exist; preserve them for inspection")
            if reuse:
                if previous.get("results") != file_identity(result_path, content=True):
                    raise ValueError(f"Existing {metric} records changed")
                records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
                summary = aggregate(records, metric, [item.utt for item in items])
                if summary != previous.get("summary"):
                    raise ValueError(f"Existing {metric} aggregation changed")
            else:
                write_json(manifest_path, {"status": "running", "contract": contract})
                records = scorers[metric]()
                if runtime_settings() != identities[metric]["execution"]:
                    raise RuntimeError("Thread or TF32 settings changed during scoring")
                summary = aggregate(records, metric, [item.utt for item in items])
                with result_path.open("x", encoding="utf-8") as handle:
                    for row in records:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                write_json(manifest_path, {"status": "complete", "contract": contract,
                                           "results": file_identity(result_path, content=True), "summary": summary})
            summaries["cer" if metric == "wer" and args.lang == "zh" else metric] = summary
    return summaries


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
