"""Leakage-aware fold orchestration for external sample embeddings."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np


class SampleEmbedding:
    """Internal K-fold utility for the Structure-aware Feature Encoder.

    Training rows are embedded with held-out folds so their target context is
    not leaked. Query rows are embedded once with the complete training
    context. The backend estimator axis is preserved here and averaged by the
    provider after all folds have been concatenated.
    """

    def __init__(
        self,
        model: Any,
        n_fold: int = 0,
        output_transform: Callable[[Any, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Configure held-out training embeddings for the backend model.

        Args:
            model: External embedding model or fitted estimator instance.
            n_fold: Held-out fold count; zero uses the complete context.
            output_transform: Optional callback applied to raw backend embeddings.
        """
        self.model = model
        self.n_fold = n_fold
        self.output_transform = output_transform
        if not hasattr(model, "get_full_embeddings"):
            raise AttributeError("The supplied model does not expose get_full_embeddings")

    def get_full_embeddings(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X: np.ndarray,
        data_source: Literal["train", "test"],
        layer: int | None = None,
    ) -> np.ndarray:
        """Extract leakage-free train embeddings or contextual query embeddings.

        Args:
            X_train: Training features that define the embedding context.
            y_train: Training target values defining the embedding context.
            X: Feature rows used by the current operation.
            data_source: Backend extraction mode, ``"train"`` or ``"test"``.
            layer: Optional zero-based encoder layer to extract.
        """
        if self.n_fold == 0:
            self.model.fit(X_train, y_train)
            result = self.model.get_full_embeddings(X, data_source=data_source, layer=layer)
            return self._transform_output(result)
        if self.n_fold < 2:
            raise ValueError("n_fold must be 0 or at least 2")
        if self.n_fold > len(X_train):
            raise ValueError("n_fold cannot exceed the number of training rows")
        from sklearn.model_selection import KFold

        if data_source == "test":
            self.model.fit(X_train, y_train)
            result = self.model.get_full_embeddings(X, data_source="test", layer=layer)
            return self._transform_output(result)
        embeddings = []
        for train_index, validation_index in KFold(self.n_fold, shuffle=False).split(X_train):
            self.model.fit(
                self._select_rows(X_train, train_index),
                self._select_rows(y_train, train_index),
            )
            embeddings.append(
                self._transform_output(
                    self.model.get_full_embeddings(
                        self._select_rows(X_train, validation_index),
                        data_source="test",
                        layer=layer,
                    )
                )
            )
        return np.concatenate(embeddings, axis=1)

    def _transform_output(self, result: np.ndarray) -> np.ndarray:
        """Apply an optional provider-owned normalization before fold concatenation.

        Args:
            result: Array or object produced by the preceding operation.
        """
        if self.output_transform is None:
            return result
        return self.output_transform(self.model, result)

    @staticmethod
    def _select_rows(values: Any, indices: np.ndarray) -> Any:
        """Select rows while preserving pandas inputs for the backend.

        Args:
            values: Values handled by the current transformation.
            indices: Original integer row positions.
        """
        return values.iloc[indices] if hasattr(values, "iloc") else values[indices]
