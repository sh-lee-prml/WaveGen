"""Small manifests, strict metadata, and exclusive output ownership."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean


@dataclass(frozen=True)
class SeedItem:
    utt: str
    ref_text: str
    ref_audio: Path
    text: str
    target_audio: Path


def read_metadata(path, dataset="seedtts", librispeech_test_clean_path=None):
    """Read prompts and target text; never use target audio or target duration to generate."""
    path = Path(path).resolve(strict=True)
    if dataset not in ("seedtts", "librispeech"):
        raise ValueError("dataset must be seedtts or librispeech")
    if dataset == "librispeech":
        if not librispeech_test_clean_path:
            raise ValueError("LibriSpeech requires --librispeech-test-clean-path")
        root = Path(librispeech_test_clean_path).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Expected a LibriSpeech test-clean directory: {root}")
    items, seen = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if dataset == "librispeech":
            fields = line.split("\t")
            if len(fields) != 6:
                raise ValueError(f"{path.name}:{line_number}: expected 6 tab-separated fields")
            ref_id, _, ref_text, utt, _, text = fields
            if not all(re.fullmatch(r"\d+-\d+-\d+", value) for value in (ref_id, utt)):
                raise ValueError(f"Invalid LibriSpeech reference/target ID at {path.name}:{line_number}")
            ref_audio = str(root.joinpath(*ref_id.split("-")[:2], f"{ref_id}.flac"))
            target = root.joinpath(*utt.split("-")[:2], f"{utt}.flac")
        else:
            fields = line.strip().split("|")
            if len(fields) not in (4, 5):
                raise ValueError(f"{path.name}:{line_number}: expected 4 or 5 pipe-separated fields")
            utt, ref_text, ref_audio, text = fields[:4]
            target = Path(fields[4]) if len(fields) == 5 else Path("wavs") / f"{utt}.wav"
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}
        if (not re.fullmatch(r"[\w.-]+", utt) or utt in (".", "..")
                or utt.endswith(".") or utt.split(".")[0].upper() in reserved):
            raise ValueError(f"Unsafe utterance ID: {utt!r}")
        if utt.casefold() in seen:
            raise ValueError(f"Duplicate utterance ID: {utt}")
        if not ref_text.strip() or not text.strip() or not ref_audio.strip():
            raise ValueError(f"Empty text or reference path for {utt}")
        seen.add(utt.casefold())
        ref = Path(ref_audio)
        ref = ref if ref.is_absolute() else path.parent / ref
        target = target if target.is_absolute() else path.parent / target
        items.append(SeedItem(utt, ref_text, ref.resolve(strict=True), text, target.resolve()))
    if not items:
        raise ValueError("Metadata contains no utterances")
    return items


def file_identity(path, *, content=False):
    """Large assets use path/stat identity, not an implied full-content proof."""
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    stat = path.stat()
    value = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        value["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {p.relative_to(root).as_posix(): file_identity(p, content=True)["sha256"]
            for p in sorted(root.rglob("*.py"))}


def metadata_identity(path, items):
    return {"metadata": file_identity(path, content=True), "count": len(items),
            "references": {str(item.ref_audio): file_identity(item.ref_audio)
                           for item in items}}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_settings():
    import torch

    return {"num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


@contextmanager
def output_lock(directory, name):
    path = Path(directory) / name
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Output is locked: {path}. Do not remove a live owner's lock.") from exc
    try:
        with handle:
            json.dump({"pid": os.getpid()}, handle)
        yield
    finally:
        path.unlink()


def wav_inventory(directory, items):
    directory = Path(directory)
    expected = {f"{item.utt}.wav" for item in items}
    actual = {p.name for p in directory.glob("*.wav")}
    if actual != expected:
        raise ValueError(f"WAV coverage mismatch: missing={sorted(expected-actual)[:5]}, "
                         f"extra={sorted(actual-expected)[:5]}")
    return {name: file_identity(directory / name) for name in sorted(expected)}


def check_generation(directory, items):
    directory = Path(directory)
    if (directory / ".inference.lock").exists():
        raise RuntimeError(f"Generation is still locked: {directory}")
    manifest = load_json(directory / "inference_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Generation is incomplete: {directory}")
    if manifest.get("ids") != [item.utt for item in items]:
        raise ValueError("Metadata IDs/order differ from the generation manifest")
    if manifest.get("wavs") != wav_inventory(directory, items):
        raise ValueError("Generated WAV inventory changed since completion")
    return manifest


def aggregate(records, metric, expected_ids):
    if [row.get("wav") for row in records] != list(expected_ids):
        raise ValueError("Metric IDs/order do not match the complete metadata")
    values = [float(row[metric]) for row in records]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Metric results must be nonempty and finite")
    if metric == "wer" and any(x < 0 for x in values):
        raise ValueError("WER cannot be negative")
    if metric == "sim" and any(not -1.00001 <= x <= 1.00001 for x in values):
        raise ValueError("Cosine similarity is outside [-1, 1]")
    mean = fmean(values)
    return {"count": len(values), "mean": mean,
            "percent": mean * 100 if metric == "wer" else None}
