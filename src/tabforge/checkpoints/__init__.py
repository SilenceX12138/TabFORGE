"""Canonical checkpoints and official Hugging Face backbone resolution."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from .canonical import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    build_checkpoint,
    detokeniser_compatibility_reason,
    is_detokeniser_compatible,
    load_checkpoint,
    save_checkpoint,
    schema_fingerprint,
    schema_from_processor,
    validate_checkpoint,
)

_OFFICIAL_CHECKPOINT_REPOSITORY = "XiangjianJiang/TabFORGE"
_OFFICIAL_CHECKPOINT_REVISION = "db4e8bebeeda1e14b3affe20ca79a97c72bd05de"


def official_checkpoint_path() -> Path:
    """Download and return the cached official backbone checkpoint."""
    checkpoint = hf_hub_download(
        repo_id=_OFFICIAL_CHECKPOINT_REPOSITORY,
        filename=CHECKPOINT_FILENAME,
        revision=_OFFICIAL_CHECKPOINT_REVISION,
        library_name="tabforge",
    )

    return Path(checkpoint)


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "build_checkpoint",
    "detokeniser_compatibility_reason",
    "official_checkpoint_path",
    "is_detokeniser_compatible",
    "load_checkpoint",
    "save_checkpoint",
    "schema_fingerprint",
    "schema_from_processor",
    "validate_checkpoint",
]
