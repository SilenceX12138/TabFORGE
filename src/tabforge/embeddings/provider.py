"""Embedding-provider abstraction and the transient TabPFN implementation.

The provider retains fitted tabular context and CPU arrays while loading the
vendored TabPFN runtime only for extraction. Encoder checkpoint bytes and live
encoder models stay outside fitted TabFORGE state.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from tabforge.config import TabFORGEEmbeddingConfig


class EmbeddingCapabilityError(RuntimeError):
    """Raised when an estimator cannot prepare or use its embedding provider."""


class EmbeddingProvider(ABC):
    """Small provider interface used by the estimator layer."""

    @abstractmethod
    def fit(self, X: Any, y: Any, *, task: str, processor: Any) -> np.ndarray:
        """Fit an embedding context and return training embeddings.

        Args:
            X: Raw training features.
            y: Optional training target.
            task: Classification, regression, or unsupervision.
            processor: Fitted feature processor.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: Any, *, processor: Any, layer: int | None = None) -> np.ndarray:
        """Transform rows with the fitted embedding context.

        Args:
            X: Raw query features.
            processor: Fitted feature processor.
            layer: Optional zero-based encoder layer to extract.
        """
        raise NotImplementedError


class TabPFNEmbeddingProvider(EmbeddingProvider):
    """Structure-aware Feature Encoder for full per-feature latent grids.

    This provider owns the TabPFN full per-feature embedding stage.  It keeps
    the backend model transient, returns a CPU-owned grid with shape
    ``(rows, feature_tokens, embedding_dimension)``, and stores only the raw
    training context required to embed future query rows.

    ``model_path='mock'`` is an explicit deterministic backend useful for unit
    tests and CPU-only examples. Production configurations use ``'auto'`` or a
    local checkpoint path and therefore resolve the pinned official encoder.
    """

    RUNTIME_COMPATIBILITY_ID = "tabpfn-v2.5"
    MAX_CLASSIFICATION_CLASSES = 10

    def __init__(
        self,
        config: TabFORGEEmbeddingConfig | None = None,
        *,
        random_state: int | None = None,
        categorical_features: tuple[int | str, ...] | None = None,
        device: str,
    ) -> None:
        """Configure a transient TabPFN embedding provider.

        Args:
            config: External encoder extraction configuration.
            random_state: Seed for mock embeddings and backend ensembles.
            categorical_features: Raw categorical schema positions or names.
            device: Device used by the transient encoder runtime.
        """
        self.config = config or TabFORGEEmbeddingConfig()
        self.random_state = random_state
        self.categorical_features = categorical_features
        self.device = device

    def fit(self, X: Any, y: Any, *, task: str, processor: Any) -> np.ndarray:
        """Fit the embedding context and return training token grids.

        Args:
            X: Raw training features.
            y: Optional training target.
            task: Classification, regression, or unsupervision.
            processor: Fitted feature processor.
        """
        backend = self._prepare_fit(X, y, task, processor)
        result = self._fit_embeddings(backend["X"], backend["y"], task, processor)
        result = self._normalize_training_output(result, len(X), task)
        return self.finalize_fit(result)

    def fit_shard(
        self,
        X: Any,
        y: Any,
        *,
        task: str,
        processor: Any,
        rank: int,
        world_size: int,
    ) -> dict[str, np.ndarray]:
        """Fit shared context and extract this rank's training embeddings.

        Args:
            X: Complete raw training features.
            y: Optional complete training target.
            task: Classification, regression, or unsupervision.
            processor: Fitted feature processor.
            rank: Current distributed rank.
            world_size: Number of participating ranks.
        """
        backend = self._prepare_fit(X, y, task, processor)
        indices = np.arange(rank, len(X), world_size, dtype=np.int64)
        if self._is_mock_backend():
            local_X = self._select_indices(X, indices)
            local_y = self._select_indices(backend["y"], indices)
            result = self._mock_embeddings(local_X, local_y, processor, task=task, target_encoded=True)
            if task == "unsupervision":
                result = result[:, : self._n_features_, :]
        elif self.config.n_folds >= 2:
            indices, result = self._fit_fold_shard(backend["X"], backend["y"], task, rank, world_size)
        else:
            local_X = self._select_indices(backend["X"], indices)
            result = self._backend_embeddings(backend["X"], backend["y"], local_X, data_source="train")
            result = self._normalize_training_output(result, len(indices), task)
        return {
            "indices": indices,
            "embeddings": np.ascontiguousarray(result, dtype=np.float32),
        }

    def _prepare_fit(self, X: Any, y: Any, task: str, processor: Any) -> dict[str, Any]:
        """Record fitted context and prepare the external backend inputs.

        Args:
            X: Raw training features.
            y: Optional training target.
            task: Validated estimator task.
            processor: Fitted feature processor.
        """
        if task not in {"classification", "regression", "unsupervision"}:
            raise ValueError(f"Unsupported embedding task {task!r}")
        if len(X) == 0:
            raise ValueError("At least one row is required to compute embeddings")
        self._record_fit_context(X, y, task, processor)
        backend_X = self._embedding_input(X, processor)
        backend_y = self._embedding_target(backend_X, y, task, processor)
        # Select the external encoder from target cardinality.
        self.encoder_task_ = task if task in {"classification", "regression"} else self._infer_encoder_task(backend_y)
        if self._is_mock_backend():
            self.encoder_layer_count_ = 8
        self._backend_y_ = self._prepare_backend_target(backend_y)
        self._train_backend_X_ = self._copy_input(backend_X)
        return {"X": backend_X, "y": self._backend_y_}

    def _normalize_training_output(self, result: Any, rows: int, task: str) -> np.ndarray:
        """Normalize backend axes and remove the unsupervised pseudo-target token.

        Args:
            result: Raw external encoder output.
            rows: Expected output row count.
            task: Fitted estimator task.
        """
        result = self._normalize_output(result, n_rows=rows, n_tokens=self._expected_token_count(task))
        if task == "unsupervision":
            result = result[:, : self._n_features_, :]
        return result

    def finalize_fit(self, result: np.ndarray) -> np.ndarray:
        """Store a complete training grid after local or distributed extraction.

        Args:
            result: Complete normalized training embedding grid.
        """
        self.embedding_shape_ = tuple(result.shape[1:])
        self.embedding_dimension_ = int(result.shape[-1])
        self._train_embeddings_ = np.ascontiguousarray(result, dtype=np.float32)
        return self._train_embeddings_.copy()

    def _fit_fold_shard(
        self,
        X: Any,
        target: np.ndarray,
        task: str,
        rank: int,
        world_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract each fold's rank-local validation rows without target leakage.

        Args:
            X: Backend training features.
            target: Backend-compatible target values.
            task: Fitted estimator task.
            rank: Current distributed rank.
            world_size: Number of participating ranks.
        """
        from sklearn.model_selection import KFold

        folds = min(self.config.n_folds, len(X))
        classifier, regressor, inference_type, preprocessor_type, extractor_type = self._backend_runtime()
        model_cls = classifier if self.encoder_task_ == "classification" else regressor
        model, extractor = self._build_backend_extractor(
            model_cls, inference_type, preprocessor_type, extractor_type, X, n_folds=0
        )
        indices_all = []
        embeddings_all = []
        try:
            for train_indices, valid_indices in KFold(folds, shuffle=False).split(X):
                local_indices = valid_indices[rank::world_size]
                if len(local_indices) == 0:
                    continue
                model.fit(
                    self._select_indices(X, train_indices),
                    self._select_indices(target, train_indices),
                )
                self._record_encoder_layer_count(model)
                raw = model.get_full_embeddings(
                    self._select_indices(X, local_indices),
                    data_source="test",
                    layer=self.config.layer,
                )
                if task != "unsupervision":
                    raw = self._restore_supervised_feature_tokens(model, raw)
                embeddings_all.append(self._normalize_training_output(raw, len(local_indices), task))
                indices_all.append(local_indices)
        finally:
            extractor.model = None
        return np.concatenate(indices_all), np.concatenate(embeddings_all, axis=0)

    def _record_fit_context(self, X: Any, y: Any, task: str, processor: Any) -> None:
        """Retain the fitted table context required for query embeddings.

        Args:
            X: Raw training features.
            y: Optional training target.
            task: Fitted estimator task.
            processor: Fitted feature processor.
        """
        self.task_ = task
        self._processor_schema_ = processor.schema_dict()
        self._train_X_ = self._copy_input(X)
        self._train_y_ = None if y is None else np.asarray(y).copy()
        self._n_features_ = processor.n_features_in_
        self._include_target_token_ = task != "unsupervision"

    def _embedding_target(self, X: Any, y: Any, task: str, processor: Any) -> np.ndarray:
        """Supply a deterministic pseudo-target for unsupervision embedding fits.

        Args:
            X: Backend training features.
            y: Optional supervised target.
            task: Fitted estimator task.
            processor: Fitted feature processor.
        """
        if task == "unsupervision":
            self._pseudo_target_ = (
                self._pseudo_target(X, processor) if self._is_mock_backend() else np.asarray(X)[:, -1]
            )
            return self._pseudo_target_
        return np.asarray(y)

    def _embedding_input(self, X: Any, processor: Any) -> Any:
        """Build the external backend view while preserving mock semantics.

        Args:
            X: Raw feature table.
            processor: Fitted feature processor.
        """
        if self._is_mock_backend():
            return X
        return processor.embedding_values(X)

    def _prepare_backend_target(self, target: np.ndarray) -> np.ndarray:
        """Prepare classifier labels and regression values for the backend.

        Args:
            target: Raw or pseudo-target values.
        """
        if self.encoder_task_ == "classification":
            encoded = np.unique(target, return_inverse=True)[1]
            if encoded.max(initial=-1) >= self.MAX_CLASSIFICATION_CLASSES:
                # Keep the classifier checkpoint usable for representation-only
                # high-cardinality tasks while preserving its 24-layer depth.
                encoded = encoded % self.MAX_CLASSIFICATION_CLASSES
            return encoded.astype(np.int64)
        return np.asarray(target, dtype=np.float32)

    def _fit_embeddings(self, X: Any, target: np.ndarray, task: str, processor: Any) -> np.ndarray:
        """Extract training embeddings through the configured backend.

        Args:
            X: Backend training features.
            target: Backend-compatible target values.
            task: Fitted estimator task.
            processor: Fitted feature processor.
        """
        if self._is_mock_backend():
            return self._mock_embeddings(X, target, processor, task=task, target_encoded=True)
        return self._backend_embeddings(X, target, X, data_source="train")

    def _expected_token_count(self, task: str) -> int:
        """Compute the fitted token grid width.

        Args:
            task: Fitted estimator task.
        """
        if self._is_mock_backend():
            return self._n_features_ + int(task != "unsupervision")
        return self._n_features_ + 1

    def transform(self, X: Any, *, processor: Any, layer: int | None = None) -> np.ndarray:
        """Embed query rows against the fitted training context.

        Args:
            X: Raw query feature table.
            processor: Fitted feature processor.
            layer: Optional zero-based encoder layer. ``None`` uses the layer
                configured for the fitted TabFORGE latent space.
        """
        if not hasattr(self, "task_"):
            raise RuntimeError("Embedding provider has not been fitted")
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int) or layer < 0):
            raise ValueError("layer must be a non-negative integer or None")
        if layer is not None and layer >= self.encoder_layer_count():
            raise ValueError(f"encoder layer must be between 0 and {self.encoder_layer_count() - 1}")
        result = self._query_embeddings(X, processor, layer=layer)
        result = self._normalize_output(result, n_rows=len(X), n_tokens=self._expected_token_count(self.task_))
        if self.task_ == "unsupervision":
            result = result[:, : self._n_features_, :]
        if tuple(result.shape[1:]) != tuple(self.embedding_shape_):
            raise EmbeddingCapabilityError(
                f"Embedding shape changed from {self.embedding_shape_} to {tuple(result.shape[1:])}"
            )
        return np.ascontiguousarray(result, dtype=np.float32)

    def _query_embeddings(self, X: Any, processor: Any, *, layer: int | None) -> np.ndarray:
        """Extract query embeddings from the configured backend.

        Args:
            X: Raw query feature table.
            processor: Fitted feature processor.
            layer: Optional zero-based encoder layer.
        """
        if self._is_mock_backend():
            target = (
                self._pseudo_target(X, processor)
                if self.task_ == "unsupervision"
                else np.zeros(len(X), dtype=np.float32)
            )
            result = self._mock_embeddings(X, target, processor, task=self.task_, target_encoded=True)
        else:
            query = self._embedding_input(X, processor)
            result = self._backend_embeddings(
                self._train_backend_X_,
                self._backend_y_,
                query,
                data_source="test",
                layer=layer,
            )
        return result

    def train_embeddings(self) -> np.ndarray:
        """Return a copy of the fitted training embeddings."""
        if not hasattr(self, "_train_embeddings_"):
            raise RuntimeError("Embedding provider has not been fitted")
        return self._train_embeddings_.copy()

    def encoder_layer_count(self) -> int:
        """Return the physical encoder depth recorded from the fitted backend."""
        if not hasattr(self, "encoder_layer_count_"):
            raise RuntimeError("Embedding provider has no recorded encoder depth")
        return int(self.encoder_layer_count_)

    def context_dict(self) -> dict[str, Any]:
        """Serialize provider context without the transient backend model."""
        if not hasattr(self, "task_"):
            raise RuntimeError("Embedding provider has not been fitted")
        return {
            "runtime_compatibility_id": self.RUNTIME_COMPATIBILITY_ID,
            "config": self.config.to_dict(),
            "random_state": self.random_state,
            "categorical_features": self.categorical_features,
            "task": self.task_,
            "encoder_task": self.encoder_task_,
            "encoder_layer_count": self.encoder_layer_count(),
            "train_X": self._train_X_,
            "train_y": self._train_y_,
            "train_backend_X": self._train_backend_X_,
            "backend_y": self._backend_y_,
            "pseudo_target": getattr(self, "_pseudo_target_", None),
            "embedding_shape": tuple(self.embedding_shape_),
            "embedding_dimension": self.embedding_dimension_,
            "n_features": self._n_features_,
            "requested_dimension": getattr(self, "_requested_dimension", None),
        }

    @classmethod
    def from_context(cls, context: dict[str, Any], *, device: str) -> "TabPFNEmbeddingProvider":
        """Restore provider context for fitted checkpoint inference.

        Args:
            context: Serialized provider state without a live encoder.
            device: Device for future transient encoder extraction.
        """
        if context.get("runtime_compatibility_id") != cls.RUNTIME_COMPATIBILITY_ID:
            raise EmbeddingCapabilityError("The embedding provider context is incompatible with this TabFORGE runtime")
        provider = cls(
            TabFORGEEmbeddingConfig(**context["config"]),
            random_state=context.get("random_state"),
            categorical_features=(
                tuple(context["categorical_features"]) if context.get("categorical_features") is not None else None
            ),
            device=device,
        )
        provider.task_ = context["task"]
        provider.encoder_task_ = context.get("encoder_task")
        if provider.encoder_task_ is None:
            target = context.get("pseudo_target") if provider.task_ == "unsupervision" else context.get("train_y")
            provider.encoder_task_ = provider._infer_encoder_task(target)
        provider._train_X_ = context["train_X"]
        provider._train_y_ = context.get("train_y")
        provider._train_backend_X_ = context["train_backend_X"]
        provider._backend_y_ = context["backend_y"]
        provider._pseudo_target_ = context.get("pseudo_target")
        provider.embedding_shape_ = tuple(context["embedding_shape"])
        provider.embedding_dimension_ = int(context["embedding_dimension"])
        if context.get("requested_dimension") is not None:
            provider._requested_dimension = int(context["requested_dimension"])
        provider._n_features_ = int(context["n_features"])
        provider._include_target_token_ = provider.task_ != "unsupervision"
        depth = context.get("encoder_layer_count")
        provider.encoder_layer_count_ = (
            (
                8
                if provider._is_mock_backend()
                else checkpoint_encoder_depth(provider.config.model_path, provider.encoder_task_)
            )
            if depth is None
            else int(depth)
        )
        return provider

    def _backend_embeddings(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X: Any,
        *,
        data_source: str,
        layer: int | None = None,
    ) -> np.ndarray:
        """Extract embeddings through the transient vendored TabPFN runtime.

        Args:
            X_train: Backend training context features.
            y_train: Backend-compatible training targets.
            X: Rows to embed.
            data_source: Backend ``"train"`` or ``"test"`` extraction mode.
            layer: Optional call-specific encoder layer.
        """
        classifier, regressor, inference_type, preprocessor_type, extractor_type = self._backend_runtime()
        model_cls = classifier if self.encoder_task_ == "classification" else regressor
        model = None
        extractor = None
        try:
            model, extractor = self._build_backend_extractor(
                model_cls, inference_type, preprocessor_type, extractor_type, X_train
            )
            result = self._extract_backend_embeddings(extractor, X_train, y_train, X, data_source, layer)
            self._record_encoder_layer_count(model)
            return result
        finally:
            # Keep the external encoder outside fitted provider state.
            if extractor is not None:
                extractor.model = None
            del extractor
            del model

    def _record_encoder_layer_count(self, model: Any) -> None:
        """Record and validate the common depth of live TabPFN architectures."""
        architectures = getattr(model, "models_", None)
        if not architectures:
            raise EmbeddingCapabilityError("Fitted TabPFN backend does not expose model.models_")
        depths = [len(architecture.transformer_encoder.layers) for architecture in architectures]
        if len(set(depths)) != 1:
            raise EmbeddingCapabilityError("TabPFN ensemble members expose inconsistent encoder depths")
        self.encoder_layer_count_ = int(depths[0])

    @staticmethod
    def _backend_runtime() -> tuple[type, type, type, type, type]:
        """Load vendored backend types only when real embeddings are requested."""
        try:
            from tabforge._vendor.tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabforge._vendor.tabpfn.inference_config import InferenceConfig, PreprocessorConfig
            from tabforge.embeddings.sample_embedding import SampleEmbedding
        except ImportError as exc:  # pragma: no cover - package data error
            raise EmbeddingCapabilityError("The vendored TabPFN runtime is unavailable") from exc
        return (
            TabPFNClassifier,
            TabPFNRegressor,
            InferenceConfig,
            PreprocessorConfig,
            SampleEmbedding,
        )

    def _build_backend_extractor(
        self,
        model_cls: Any,
        inference_type: Any,
        preprocessor_type: Any,
        extractor_type: Any,
        X_train: Any,
        n_folds: int | None = None,
    ) -> tuple[Any, Any]:
        """Construct the vendored embedding extractor and its inference model.

        Args:
            model_cls: Vendored classifier or regressor class.
            inference_type: Vendored inference configuration class.
            preprocessor_type: Vendored preprocessor configuration class.
            extractor_type: Fold-aware embedding extractor class.
            X_train: Backend training features.
            n_folds: Optional fold-count override.
        """
        checkpoint = resolve_checkpoint(self.config.model_path, self.encoder_task_)
        inference_config = inference_type(
            FINGERPRINT_FEATURE=False,
            FEATURE_SHIFT_METHOD=None,
            CLASS_SHIFT_METHOD=None,
            PREPROCESS_TRANSFORMS=[preprocessor_type(name="none")],
        )
        model = model_cls(
            n_estimators=self.config.n_estimators,
            model_path=checkpoint,
            device=self.device,
            random_state=self.random_state,
            inference_config=inference_config,
            ignore_pretraining_limits=True,
        )
        folds = self.config.n_folds if n_folds is None else n_folds
        if folds and len(X_train) < folds:
            folds = len(X_train) if len(X_train) >= 2 else 0
        # TabPFN preprocessing can remove fold-constant columns for any task.
        # Restore their token slots before combining uneven held-out folds.
        output_transform = self._restore_supervised_feature_tokens
        return model, extractor_type(model=model, n_fold=folds, output_transform=output_transform)

    def _restore_supervised_feature_tokens(self, model: Any, result: np.ndarray) -> np.ndarray:
        """Restore feature slots dropped by a supervised backend fit.

        Args:
            model: Fitted transient backend model with preprocessing metadata.
            result: Raw estimator-stacked embedding output.
        """
        array = np.asarray(result)
        expected_tokens = self._n_features_ + 1
        if array.ndim != 4 or array.shape[2] == expected_tokens:
            return array
        configs = model.executor_.ensemble_configs
        preprocessors = model.executor_.preprocessors
        if len(array) != len(configs) or len(array) != len(preprocessors):
            raise EmbeddingCapabilityError("Backend estimator metadata does not match its embedding output")

        restored = np.zeros((*array.shape[:2], expected_tokens, array.shape[3]), dtype=array.dtype)
        for estimator_index, (embedding, config, steps) in enumerate(zip(array, configs, preprocessors)):
            source_indices = np.arange(self._n_features_)
            for step in steps:
                selection = getattr(step, "sel_", None)
                if selection is not None:
                    source_indices = source_indices[np.asarray(selection, dtype=bool)]
                permutation = getattr(step, "index_permutation_", None)
                if permutation is not None:
                    source_indices = source_indices[np.asarray(permutation, dtype=np.int64)]

            architecture = model.models_[config._model_index]
            for module in architecture.modules():
                selection = getattr(module, "column_selection_mask", None)
                if selection is not None:
                    source_indices = source_indices[np.asarray(selection.detach().cpu()).reshape(-1, order="C")]

            if embedding.shape[1] != len(source_indices) + 1:
                raise EmbeddingCapabilityError(
                    "Cannot align backend feature tokens with the original supervised schema"
                )
            restored[estimator_index][:, source_indices, :] = embedding[:, :-1, :]
            restored[estimator_index, :, -1, :] = embedding[:, -1, :]
        return restored

    @staticmethod
    def _select_indices(values: Any, indices: np.ndarray) -> Any:
        """Select arbitrary rows while preserving pandas inputs.

        Args:
            values: Source pandas or NumPy values.
            indices: Integer row positions.
        """
        return values.iloc[indices] if hasattr(values, "iloc") else np.asarray(values)[indices]

    def _extract_backend_embeddings(
        self,
        extractor: Any,
        X_train: Any,
        y_train: np.ndarray,
        X: Any,
        data_source: str,
        layer: int | None,
    ) -> np.ndarray:
        """Run backend extraction and normalize fold and model axes.

        Args:
            extractor: Configured fold-aware backend extractor.
            X_train: Backend training context features.
            y_train: Backend-compatible training targets.
            X: Rows to embed.
            data_source: Backend train/test extraction mode.
            layer: Optional call-specific encoder layer.
        """
        query_batches = self._backend_query_batches(X, data_source)
        extracted = []
        for query in query_batches:
            raw = extractor.get_full_embeddings(
                self._as_numpy_or_frame(X_train),
                np.asarray(y_train),
                self._as_numpy_or_frame(query),
                data_source=data_source,
                layer=self.config.layer if layer is None else layer,
            )
            if len(query_batches) > 1:
                raw = self._normalize_output(raw, n_rows=len(query), n_tokens=self._n_features_ + 1)
            extracted.append(raw)
        return extracted[0] if len(extracted) == 1 else np.concatenate(extracted, axis=0)

    def _backend_query_batches(self, X: Any, data_source: str) -> list[Any]:
        """Split test queries according to the configured backend batch size.

        Args:
            X: Backend query rows.
            data_source: Backend train/test extraction mode.
        """
        if data_source != "test" or self.config.batch_size is None:
            return [X]
        return [
            self._select_rows(X, start, min(start + self.config.batch_size, len(X)))
            for start in range(0, len(X), self.config.batch_size)
        ]

    def _mock_embeddings(
        self,
        X: Any,
        y: np.ndarray | None,
        processor: Any,
        *,
        task: str,
        target_encoded: bool = False,
    ) -> np.ndarray:
        """Create deterministic embeddings for lightweight tests and examples.

        Args:
            X: Raw feature rows.
            y: Optional target or pseudo-target values.
            processor: Fitted feature processor.
            task: Fitted estimator task.
            target_encoded: Whether ``y`` is already backend encoded.
        """
        # Keep mock and real providers on the same numerical-first token contract.
        token_values = processor.embedding_values(X)
        if task != "unsupervision":
            target_values = np.asarray(y, dtype=np.float32) if target_encoded else processor.target_token_values(y)
            if task == "classification" and processor.target_cardinality_ > 1:
                target_values = target_values / (processor.target_cardinality_ - 1)
            token_values = np.concatenate([token_values, target_values[:, None]], axis=1)
        dimension = self.configured_dimension()
        seed = 0 if self.random_state is None else int(self.random_state)
        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, 1.0, size=(token_values.shape[1], dimension)).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, size=(token_values.shape[1], dimension)).astype(np.float32)
        token_ids = np.arange(token_values.shape[1], dtype=np.float32)[:, None]
        result = np.sin(token_values[:, :, None] * weights[None] + phase[None] + token_ids[None] / 11.0)
        return result.astype(np.float32)

    def configured_dimension(self) -> int:
        """Return the fitted or explicitly requested embedding dimension."""
        if hasattr(self, "embedding_dimension_"):
            return int(self.embedding_dimension_)
        configured = getattr(self, "_requested_dimension", None)
        if configured is not None:
            return configured
        # Real-TabPFN v2.5 produces 192-dimensional embeddings by default.
        return 192

    def _is_mock_backend(self) -> bool:
        """Return whether the explicit test backend is selected."""
        return str(self.config.model_path).lower() in {"mock", "synthetic", "none"}

    def _copy_input(self, X: Any) -> Any:
        """Copy table inputs without changing their container type.

        Args:
            X: Source pandas or NumPy table.
        """
        return X.copy(deep=True) if hasattr(X, "copy") and hasattr(X, "columns") else np.asarray(X).copy()

    def _as_numpy_or_frame(self, X: Any) -> Any:
        """Preserve backend-supported NumPy and pandas inputs.

        Args:
            X: Source pandas or NumPy table.
        """
        return X

    @staticmethod
    def _select_rows(values: Any, start: int, end: int) -> Any:
        """Select a contiguous row slice while preserving pandas metadata.

        Args:
            values: Source pandas or NumPy values.
            start: Inclusive row position.
            end: Exclusive row position.
        """
        return values.iloc[start:end] if hasattr(values, "iloc") else values[start:end]

    @staticmethod
    def _pseudo_target(X: Any, processor: Any) -> np.ndarray:
        """Derive unsupervised backend context from the final feature token.

        Args:
            X: Raw feature rows.
            processor: Fitted feature processor.
        """
        return processor.embedding_values(X)[:, -1]

    @staticmethod
    def _infer_encoder_task(target: np.ndarray) -> str:
        """Infer the external encoder task from target cardinality.

        Args:
            target: Backend target or pseudo-target values.
        """
        return "classification" if np.unique(target).size <= 10 else "regression"

    @staticmethod
    def _normalize_output(
        result: np.ndarray,
        *,
        n_rows: int | None = None,
        n_tokens: int | None = None,
    ) -> np.ndarray:
        """Normalize supported backend axes to ``(rows, tokens, dimension)``.

        Args:
            result: Raw backend embedding array.
            n_rows: Expected row count when known.
            n_tokens: Expected token count when known.

        The vendored backend normally returns ``(estimators, rows, tokens,
        dimension)`` and a one-estimator path may squeeze to either
        ``(rows, tokens, dimension)`` or ``(tokens, rows, dimension)``.  Axis
        decisions use the known row/token counts; ambiguous arrays fail early
        instead of being silently transposed.
        """
        array = TabPFNEmbeddingProvider._normalize_backend_rank(np.asarray(result), n_rows, n_tokens)
        if array.ndim != 3:
            raise EmbeddingCapabilityError(f"Expected a 3-D embedding grid, got {array.shape}")
        if n_rows is None:
            return array
        array = TabPFNEmbeddingProvider._normalize_row_axis(array, n_rows, n_tokens)
        if n_tokens is not None and array.shape[1] != n_tokens:
            raise EmbeddingCapabilityError(
                f"Expected {n_tokens} embedding tokens after axis normalization, got {array.shape}"
            )
        return array

    @staticmethod
    def _normalize_backend_rank(array: np.ndarray, n_rows: int | None, n_tokens: int | None) -> np.ndarray:
        """Reduce estimator axes and restore squeezed single-row outputs.

        Args:
            array: Raw backend embedding array.
            n_rows: Expected row count when known.
            n_tokens: Expected token count when known.
        """
        if array.ndim == 4:
            return TabPFNEmbeddingProvider._reduce_estimator_axis(array, n_rows, n_tokens)
        if array.ndim != 2 or n_rows != 1 or n_tokens is None:
            return array
        if array.shape[0] == n_tokens:
            return array.reshape(1, n_tokens, array.shape[1])
        if array.shape[1] == n_tokens:
            return array.reshape(1, array.shape[0], n_tokens).transpose(0, 2, 1)
        return array

    @staticmethod
    def _reduce_estimator_axis(array: np.ndarray, n_rows: int | None, n_tokens: int | None) -> np.ndarray:
        """Identify and average the estimator axis of a four-dimensional output.

        Args:
            array: Four-dimensional backend output.
            n_rows: Expected row count.
            n_tokens: Expected token count when known.
        """
        if n_rows is None:
            raise EmbeddingCapabilityError("n_rows is required to normalize a four-dimensional embedding output")
        row_candidates = [axis for axis in range(3) if array.shape[axis] == n_rows]
        if len(row_candidates) != 1 and n_tokens is not None:
            row_candidates = [
                axis
                for axis in row_candidates
                if any(array.shape[token_axis] == n_tokens for token_axis in range(3) if token_axis != axis)
            ]
        if len(row_candidates) != 1:
            raise EmbeddingCapabilityError(
                f"Cannot identify the row axis in embedding output {array.shape}; expected {n_rows} rows"
            )
        row_axis = row_candidates[0]
        token_axes = [axis for axis in range(3) if axis != row_axis and array.shape[axis] == n_tokens]
        if len(token_axes) != 1:
            raise EmbeddingCapabilityError(
                f"Cannot identify the token axis in embedding output {array.shape}; expected {n_tokens} tokens"
            )
        estimator_axis = next(axis for axis in range(3) if axis not in {row_axis, token_axes[0]})
        return array.mean(axis=estimator_axis)

    @staticmethod
    def _normalize_row_axis(array: np.ndarray, n_rows: int, n_tokens: int | None) -> np.ndarray:
        """Move the uniquely identified row axis to the leading position.

        Args:
            array: Three-dimensional backend output.
            n_rows: Expected row count.
            n_tokens: Expected token count when known.
        """
        row_axes = [axis for axis in (0, 1) if array.shape[axis] == n_rows]
        if len(row_axes) != 1:
            if len(row_axes) == 2 and n_tokens in {array.shape[0], array.shape[1]} and array.shape[0] != array.shape[1]:
                row_axes = [0 if n_tokens == array.shape[1] else 1]
            else:
                raise EmbeddingCapabilityError(
                    f"Cannot identify the row axis in embedding output {array.shape}; expected {n_rows} rows"
                )
        return array.transpose(1, 0, 2) if row_axes[0] == 1 else array


def load_vendor_lock() -> dict[str, Any]:
    """Load immutable external checkpoint references bundled with TabFORGE."""
    package_path = Path(__file__).resolve().parents[0] / "vendor-lock.json"
    if not package_path.is_file():
        raise EmbeddingCapabilityError(f"Packaged vendor-lock.json is missing: {package_path}")
    return json.loads(package_path.read_text(encoding="utf-8"))


def resolve_checkpoint(model_path: str, task: str) -> str:
    """Resolve an explicit or pinned encoder checkpoint reference.

    Args:
        model_path: ``"auto"`` or an explicit local checkpoint path.
        task: Encoder task used to select the pinned checkpoint.
    """
    if model_path != "auto":
        path = Path(model_path)
        if not path.is_file():
            raise EmbeddingCapabilityError(f"TabPFN checkpoint does not exist: {path}")
        return str(path)
    if task == "unsupervision":
        task = "regression"
    lock = load_vendor_lock()
    checkpoint = lock["checkpoints"].get(task)
    if checkpoint is None:
        raise EmbeddingCapabilityError(f"No pinned TabPFN checkpoint mapping for task {task!r}")
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=checkpoint["repository"],
            filename=checkpoint["filename"],
            revision=checkpoint["revision"],
        )
    except Exception as exc:
        raise EmbeddingCapabilityError(
            f"Unable to resolve the pinned TabPFN {task} checkpoint {checkpoint['filename']!r}"
        ) from exc


def checkpoint_encoder_depth(model_path: str, task: str) -> int:
    """Read and validate the physical encoder depth from a TabPFN checkpoint."""
    import torch

    checkpoint_path = resolve_checkpoint(model_path, task)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        depth = int(payload["config"]["nlayers"])
        state = payload["state_dict"]
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingCapabilityError("TabPFN checkpoint is missing config.nlayers or state_dict") from exc
    prefix = "transformer_encoder.layers."
    indices = {int(key[len(prefix) :].split(".", maxsplit=1)[0]) for key in state if key.startswith(prefix)}
    if indices != set(range(depth)):
        raise EmbeddingCapabilityError("TabPFN checkpoint encoder state keys do not match config.nlayers")
    return depth
