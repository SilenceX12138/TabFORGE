"""Versioned TabFORGE checkpoint envelopes and schema compatibility helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_FORMAT = "tabforge-checkpoint-v1"
CHECKPOINT_FILENAME = "checkpoint.pt"


def schema_fingerprint(schema: Mapping[str, Any]) -> str:
    """Return a deterministic fingerprint for the detokeniser structure.

    Args:
        schema: Fitted checkpoint schema mapping, when available.
    """
    normalized = _schema_identity(schema)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_detokeniser_compatible(
    checkpoint_schema: Mapping[str, Any] | None,
    current_schema: Mapping[str, Any] | None,
) -> bool:
    """Return whether schema-dependent detokeniser state can be reused.

    Args:
        checkpoint_schema: Schema serialized in the source checkpoint.
        current_schema: Schema fitted by the current estimator.
    """
    return detokeniser_compatibility_reason(checkpoint_schema, current_schema) is None


def detokeniser_compatibility_reason(
    checkpoint_schema: Mapping[str, Any] | None,
    current_schema: Mapping[str, Any] | None,
) -> str | None:
    """Explain the first structural mismatch between two fitted schemas.

    Args:
        checkpoint_schema: Schema serialized in the source checkpoint.
        current_schema: Schema fitted by the current estimator.
    """
    if not checkpoint_schema:
        return "checkpoint does not contain detokeniser schema metadata"
    if not current_schema:
        return "current model does not contain schema metadata"
    saved = _schema_identity(checkpoint_schema)
    current = _schema_identity(current_schema)
    for key in (
        "n_numerical_features",
        "n_categorical_features",
        "feature_types",
        "categorical_cardinalities",
        "model_feature_order",
        "target_type",
        "target_representation",
    ):
        if saved[key] != current[key]:
            return f"{key} differs (checkpoint={saved[key]!r}, current={current[key]!r})"
    return None


def build_checkpoint(
    *,
    state_dict: Mapping[str, Mapping[str, torch.Tensor]],
    schema: Mapping[str, Any] | None = None,
    preprocessing: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    model_version: str = "unknown",
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate the current canonical checkpoint payload.

    Args:
        state_dict: Named component tensor mappings.
        schema: Fitted checkpoint schema mapping, when available.
        preprocessing: Serialized fitted preprocessing state.
        config: Validated configuration or serialized configuration mapping.
        training: Serialized training metadata or resolved training configuration.
        model_version: TabFORGE model-family identifier stored in the checkpoint.
        source: Checkpoint provenance metadata.
    """
    schema_payload = dict(schema or {})
    if schema_payload:
        fingerprint = schema_fingerprint(schema_payload)
        if "fingerprint" in schema_payload and schema_payload["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint schema fingerprint does not match its structural schema")
        schema_payload["fingerprint"] = fingerprint
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "format": CHECKPOINT_FORMAT,
        "model_version": model_version,
        "state_dict": {
            component: {key: _cpu_tensor(value) for key, value in component_state.items()}
            for component, component_state in state_dict.items()
        },
        "schema": schema_payload,
        "preprocessing": dict(preprocessing or {}),
        "config": dict(config or {}),
        "training": dict(training or {}),
    }
    if source is not None:
        payload["source"] = dict(source)
    validate_checkpoint(payload)
    return payload


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> Path:
    """Save a canonical checkpoint payload to a single torch file.

    Args:
        checkpoint: Canonical checkpoint mapping handled by this operation.
        path: Checkpoint source or destination path.
    """
    validate_checkpoint(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), destination)
    return destination


def load_checkpoint(
    path: str | Path,
) -> dict[str, Any]:
    """Load the current canonical TabFORGE checkpoint format.

    Args:
        path: Checkpoint source or destination path.
    """
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / CHECKPOINT_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(f"TabFORGE checkpoint not found: {candidate}")
    if candidate.suffix != ".pt":
        raise ValueError("TabFORGE checkpoints must use the canonical .pt format")
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    validate_checkpoint(payload)
    return dict(payload)


def schema_from_processor(processor: Any, task: str) -> dict[str, Any]:
    """Create the structural schema used for detokeniser compatibility.

    Args:
        processor: Fitted feature processor defining schema and layout.
        task: Classification, regression, or unsupervision contract.
    """
    raw_schema = processor.schema_dict()
    order = list(raw_schema.get("embedding_token_raw_indices", range(processor.n_features_in_)))
    numerical = set(raw_schema.get("numerical_indices", ()))
    feature_types = ["numerical" if index in numerical else "categorical" for index in order]
    target_type = None
    target_representation: dict[str, Any] = {"kind": "none", "width": 0}
    if task != "unsupervision":
        target_type = "classification" if processor.target_is_classification_ else "regression"
        width = int(processor.target_reconstruction_dim_)
        target_representation = {"kind": target_type, "width": width}
        if target_type == "classification":
            target_representation["cardinality"] = int(processor.target_cardinality_)
    schema = {
        "feature_types": feature_types,
        "categorical_cardinalities": list(raw_schema.get("categorical_cardinalities", ())),
        "model_feature_order": order,
        "target_type": target_type,
        "target_representation": target_representation,
        "n_numerical_features": len(raw_schema.get("numerical_indices", ())),
        "n_categorical_features": len(raw_schema.get("categorical_indices", ())),
        "feature_schema": raw_schema,
    }
    schema["fingerprint"] = schema_fingerprint(schema)
    return schema


def validate_checkpoint(payload: Mapping[str, Any]) -> None:
    """Validate the small structural contract understood by current TabFORGE.

    Args:
        payload: Canonical checkpoint payload.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("TabFORGE checkpoint must be a mapping")
    required = {
        "format_version",
        "format",
        "model_version",
        "state_dict",
        "schema",
        "preprocessing",
        "config",
        "training",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"TabFORGE checkpoint is missing: {', '.join(missing)}")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported TabFORGE checkpoint format: {payload.get('format_version')!r}")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported TabFORGE checkpoint format name: {payload.get('format')!r}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("TabFORGE checkpoint state_dict must be a component mapping")
    components = {"decoder", "diffusion", "detokeniser"}
    if set(state) != components:
        raise ValueError("TabFORGE checkpoint state_dict must contain decoder, diffusion, and detokeniser")
    for component, component_state in state.items():
        if not isinstance(component_state, Mapping):
            raise ValueError(f"TabFORGE checkpoint component {component!r} must be a state mapping")
        for key, value in component_state.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                raise ValueError(f"Invalid {component} checkpoint entry {key!r}")


def _schema_identity(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Select only structure that changes decoder/detokeniser meaning.

    Args:
        schema: Fitted checkpoint schema mapping, when available.
    """
    feature_types = list(schema.get("feature_types", ()))
    cardinalities = list(schema.get("categorical_cardinalities", ()))
    order = list(schema.get("model_feature_order", ()))
    target = schema.get("target_representation") or {
        "kind": schema.get("target_type"),
        "width": 0,
    }
    return {
        "n_numerical_features": int(schema.get("n_numerical_features", feature_types.count("numerical"))),
        "n_categorical_features": int(schema.get("n_categorical_features", feature_types.count("categorical"))),
        "feature_types": feature_types,
        "categorical_cardinalities": cardinalities,
        "model_feature_order": order,
        "target_type": schema.get("target_type"),
        "target_representation": {
            "kind": target.get("kind"),
            "width": int(target.get("width", 0)),
            **({"cardinality": int(target["cardinality"])} if "cardinality" in target else {}),
        },
    }


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    """Detach a tensor into contiguous CPU storage for portable serialization.

    Args:
        value: Object or tensor handled by the helper.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Checkpoint state values must be tensors, got {type(value).__name__}")
    return value.detach().cpu().contiguous()


def _json_default(value: Any) -> Any:
    """Convert schema values into JSON-compatible Python objects.

    Args:
        value: Object or tensor handled by the helper.
    """

    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "build_checkpoint",
    "detokeniser_compatibility_reason",
    "is_detokeniser_compatible",
    "load_checkpoint",
    "save_checkpoint",
    "schema_fingerprint",
    "schema_from_processor",
    "validate_checkpoint",
]
