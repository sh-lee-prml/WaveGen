"""Resolve the three inference assets from one Hugging Face model revision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re


_COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")


@dataclass(frozen=True)
class HubAssets:
    repo_id: str
    revision: str
    checkpoint: Path
    config: Path
    vocabulary: Path


def resolve_hub_assets(repo_id: str, *, revision: str,
                       checkpoint_filename="WaveGen_VSpec/model_1400000.safetensors",
                       config_filename="WaveGen_VSpec/WaveGen_VSpec.yaml",
                       vocab_filename="WaveGen_VSpec/vocab.txt", cache_dir=None,
                       local_files_only=False) -> HubAssets:
    """Fetch explicitly requested assets; never fall back to packaged files.

    Online branch/tag names resolve once to a commit before fetching all three
    files. Offline loading requires a full commit SHA and cached files.
    Authentication, when needed, follows huggingface_hub's standard settings.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must name the published Hugging Face model repository")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("An explicit revision is required; a full commit SHA is recommended")
    names = (checkpoint_filename, config_filename, vocab_filename)
    for filename in names:
        if (not isinstance(filename, str) or not filename or "\\" in filename
                or PurePosixPath(filename).is_absolute()
                or any(part in {"", ".", ".."} for part in filename.split("/"))
                or ":" in filename):
            raise ValueError("Hub filenames must be relative repository paths without '..'")
    if local_files_only and not _COMMIT.fullmatch(revision):
        raise ValueError("Offline loading requires the full 40-character commit SHA")

    from huggingface_hub import hf_hub_download, model_info

    commit = revision.lower() if _COMMIT.fullmatch(revision) else model_info(repo_id, revision=revision).sha
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("Hugging Face did not return an immutable commit SHA")
    commit = commit.lower()
    downloaded = [Path(hf_hub_download(
        repo_id=repo_id, filename=filename, revision=commit,
        cache_dir=cache_dir, local_files_only=local_files_only,
    )) for filename in names]
    return HubAssets(repo_id, commit, *downloaded)
