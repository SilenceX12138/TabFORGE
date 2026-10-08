"""Schema-aware reconstruction losses for decoded tabular values."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mixed_feature_loss(
    decoded: dict[str, torch.Tensor | list[torch.Tensor] | None],
    target: torch.Tensor,
    *,
    numerical_count: int,
    categorical_cardinalities: tuple[int, ...] | list[int],
    target_kind: str | None = None,
) -> dict[str, torch.Tensor]:
    """Compute numerical MSE and one cross-entropy term per categorical span.

    Args:
        decoded: Decoder outputs for numerical, categorical, and target heads.
        target: Dense reconstruction target matrix.
        numerical_count: Leading numerical target width.
        categorical_cardinalities: Ordered one-hot widths.
        target_kind: Classification, regression, or ``None``.
    """
    reconstruction, offset = _feature_reconstruction_loss(decoded, target, numerical_count, categorical_cardinalities)
    target_loss = _target_reconstruction_loss(decoded, target, offset, target_kind)
    return {
        "reconstruction": reconstruction,
        "target": target_loss,
        "total": reconstruction + target_loss,
    }


def _feature_reconstruction_loss(
    decoded: dict[str, torch.Tensor | list[torch.Tensor] | None],
    target: torch.Tensor,
    numerical_count: int,
    categorical_cardinalities: tuple[int, ...] | list[int],
) -> tuple[torch.Tensor, int]:
    """Accumulate numerical and categorical decoder losses.

    Args:
        decoded: Decoder output mapping.
        target: Dense reconstruction target matrix.
        numerical_count: Leading numerical target width.
        categorical_cardinalities: Ordered categorical one-hot widths.
    """
    numerical = decoded["numerical"]
    categorical = decoded["categorical"]
    offset = 0
    losses: list[torch.Tensor] = []
    if numerical_count:
        losses.append(F.mse_loss(numerical, target[:, :numerical_count], reduction="sum") / len(target))
        offset = numerical_count
    for cardinality, logits in zip(categorical_cardinalities, categorical):
        labels = target[:, offset : offset + cardinality].argmax(dim=1)
        losses.append(F.cross_entropy(logits, labels, reduction="sum") / len(target))
        offset += cardinality
    reconstruction = torch.stack(losses).sum() if losses else target.new_zeros(())
    return reconstruction, offset


def _target_reconstruction_loss(
    decoded: dict[str, torch.Tensor | list[torch.Tensor] | None],
    target: torch.Tensor,
    offset: int,
    target_kind: str | None,
) -> torch.Tensor:
    """Compute the optional classification or regression target loss.

    Args:
        decoded: Decoder output mapping.
        target: Dense reconstruction target matrix.
        offset: Start column of the target reconstruction span.
        target_kind: Classification, regression, or ``None``.
    """
    target_output = decoded.get("target")
    if target_kind is None or target_output is None:
        return target.new_zeros(())
    target_values = target[:, offset:]
    if target_kind == "classification":
        return F.cross_entropy(target_output, target_values.argmax(dim=1))
    if target_kind == "regression":
        return F.mse_loss(target_output.reshape(-1, 1), target_values[:, :1])
    return target.new_zeros(())
