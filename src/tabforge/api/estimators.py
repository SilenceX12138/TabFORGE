"""Public sklearn-style estimators for TabFORGE workflows.

Each estimator exposes one task-focused interface while reusing the fitted lifecycle
implemented by :class:`tabforge.api.base.TabFORGE`. Public masks use ``True`` for
cells to regenerate; internal diffusion masks use ``True`` for observed tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import ClassifierMixin, ClusterMixin, OutlierMixin, RegressorMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from tabforge.api.base import TabFORGE
from tabforge.config import (
    TabFORGEArchitectureConfig,
    TabFORGEDiffusionConfig,
    TabFORGEEmbeddingConfig,
    TabFORGERuntimeConfig,
    TabFORGETrainingConfig,
)
from tabforge.distributed import active as distributed_active
from tabforge.distributed import gather_objects
from tabforge.distributed import rank as distributed_rank
from tabforge.distributed import shard_indices, split_count
from tabforge.distributed import world_size as distributed_world_size
from tabforge.embeddings import EmbeddingCapabilityError


class TabFORGEGenerator(TabFORGE):
    """Generate mixed-type tables from a fitted latent reference distribution.

    ``task`` controls whether the fitted table has no target or carries a
    classification/regression target. Supervised ``generate`` calls return an
    ``(X, y)`` pair; unsupervised calls return the generated feature table.
    """

    _estimator_type_name = "generator"

    def __init__(
        self,
        *,
        task: str,
        categorical_features: Sequence[int | str] | None = None,
        embedding_config: TabFORGEEmbeddingConfig | Mapping[str, Any] | None = None,
        architecture_config: TabFORGEArchitectureConfig | Mapping[str, Any] | None = None,
        diffusion_config: TabFORGEDiffusionConfig | Mapping[str, Any] | None = None,
        training_config: TabFORGETrainingConfig | Mapping[str, Any] | None = None,
        runtime_config: TabFORGERuntimeConfig | Mapping[str, Any] | None = None,
        random_state: int | None = None,
        logger: Any = None,
        n_reference_neighbors: int = 1,
        observation_masking: bool = False,
    ) -> None:
        """Configure task-aware generation and neighbour interpolation.

        Args:
            task: ``"classification"``, ``"regression"``, or ``"unsupervision"``.
            categorical_features: Raw categorical column names or integer positions.
            embedding_config: Embedding configuration object or field mapping.
            architecture_config: Decoder and denoiser architecture configuration.
            diffusion_config: EDM training and sampling configuration.
            training_config: Optimizer, phase, masking, and validation configuration.
            runtime_config: Device, distribution, determinism, and logging options.
            random_state: Seed used for fitting and default generation calls.
            logger: Existing W&B run or compatible logger to reuse.
            n_reference_neighbors: Noisy reference trajectories averaged per row.
            observation_masking: Enable conditional observation masks during
                generator training.
        """
        super().__init__(
            categorical_features=categorical_features,
            embedding_config=embedding_config,
            architecture_config=architecture_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            runtime_config=runtime_config,
            random_state=random_state,
            logger=logger,
        )
        self.task = task
        self.n_reference_neighbors = n_reference_neighbors
        self.observation_masking = observation_masking

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any] | None = None,
        *,
        validation_data: Any = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> "TabFORGEGenerator":
        """Fit the generator for its configured task.

        Args:
            X: Training features in raw schema order.
            y: Target values for supervised tasks; ``None`` for unsupervision.
            validation_data: Validation features or an ``(X, y)`` pair matching
                the configured task.
            checkpoint: ``"pretrained"``, a canonical checkpoint path, or
                ``None`` for fresh component initialization.
            detokeniser: ``"auto"``, ``"load"``, or ``"reinitialize"``.

        Returns:
            TabFORGEGenerator: The fitted estimator.
        """
        return self._fit_with_task(
            X,
            y,
            self.task,
            validation_data=validation_data,
            checkpoint=checkpoint,
            detokeniser=detokeniser,
        )

    def generate(self, n_samples: int, *, random_state: int | None = None) -> Any:
        """Generate synthetic rows from the fitted reference distribution.

        Args:
            n_samples: Positive number of rows to generate.
            random_state: Optional seed overriding the estimator seed.

        Returns:
            Any: A feature table for unsupervision or an ``(X, y)`` pair for a
            supervised task.
        """
        # === Validate generation state ===
        self._check_fitted()
        if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(self.n_reference_neighbors, int) or self.n_reference_neighbors < 1:
            raise ValueError("n_reference_neighbors must be a positive integer")
        if not self.generation_capable_ or self.reference_embeddings_ is None:
            raise RuntimeError(
                "This checkpoint has no reference embedding bank; unconditional generation is unavailable"
            )
        if distributed_active():
            local_count = split_count(int(n_samples))[distributed_rank()]
            seed = self._generation_seed(random_state)
            local = self._generate_local(local_count, seed) if local_count else self._empty_generated()
            return self._concat_generated(gather_objects(local))
        return self._generate_local(int(n_samples), random_state)

    def _generation_seed(self, random_state: int | None) -> int:
        """Map a generation call and rank to a distinct deterministic seed.

        Args:
            random_state: Call-specific seed or ``None`` for the estimator seed.
        """
        base_seed = self._seed(random_state)
        return base_seed * distributed_world_size() + distributed_rank()

    def _generate_local(self, n_samples: int, random_state: int | None) -> Any:
        """Generate this rank's local synthetic rows.

        Args:
            n_samples: Rows assigned to the current process.
            random_state: Random seed for reference selection and diffusion.
        """
        # === Generate synthetic latents ===
        generator = self._generation_generator(random_state)
        initial = self._initial_generation_latents(n_samples, generator)
        latents = self._sample_latents(initial, generator=generator)
        # === Decode the table ===
        feature_values, target_values = self._decode(latents)
        return self._format_generated_values(feature_values, target_values)

    def _empty_generated(self) -> Any:
        """Return a schema-correct empty result for an idle rank."""
        processor = self.feature_processor_
        features = processor.inverse_transform(np.empty((0, processor.reconstruction_dim_), dtype=np.float32))
        if self.task_ == "unsupervision":
            return features
        width = processor.reconstruction_dim_ + processor.target_reconstruction_dim_
        return processor.inverse_joint(np.empty((0, width), dtype=np.float32))

    @staticmethod
    def _concat_generated(values: list[Any]) -> Any:
        """Combine rank-local generated outputs.

        Args:
            values: Rank-ordered generated tables or supervised table/target pairs.
        """
        if isinstance(values[0], tuple):
            features = pd.concat([value[0] for value in values], ignore_index=True)
            targets = np.concatenate([np.asarray(value[1]) for value in values])
            return features, targets
        return pd.concat(values, ignore_index=True)

    def _generation_generator(self, random_state: int | None) -> torch.Generator:
        """Create the shared random stream used throughout generation.

        Args:
            random_state: Call-specific seed or ``None`` for the estimator seed.
        """
        generator = torch.Generator(device=self._device_)
        generator.manual_seed(self._seed(random_state))
        return generator

    def _initial_generation_latents(self, n_samples: int, generator: torch.Generator) -> torch.Tensor:
        """Build reference, noise, and neighbour-interpolation latents.

        Args:
            n_samples: Number of latent rows to initialize.
            generator: Device-local random stream shared by the generation call.
        """
        # Reuse reference rows evenly before randomizing their order.
        bank = self.reference_embeddings_.to(self._device_)
        if n_samples < len(bank):
            indices = torch.randperm(len(bank), generator=generator, device=self._device_)[:n_samples]
        else:
            indices = torch.arange(n_samples, device=self._device_) % len(bank)
            indices = indices[torch.randperm(n_samples, generator=generator, device=self._device_)]
        clean = bank.index_select(0, indices)
        noise = torch.randn(clean.shape, generator=generator, device=self._device_)
        noisy = clean + noise * self.diffusion_config_.sigma_init
        neighbours = torch.randint(
            0,
            n_samples,
            (n_samples, self.n_reference_neighbors),
            generator=generator,
            device=self._device_,
        )
        return noisy[neighbours].mean(dim=1)

    def _format_generated_values(self, feature_values: torch.Tensor, target_values: torch.Tensor | None) -> Any:
        """Restore generated tensors to the fitted feature and target schemas.

        Args:
            feature_values: Decoded dense feature reconstruction values.
            target_values: Optional decoded target probabilities or values.
        """
        if self.task_ == "unsupervision":
            return self.feature_processor_.inverse_transform(feature_values.detach().cpu().numpy())
        X_syn, y_syn = self.feature_processor_.inverse_joint(
            torch.cat([feature_values, target_values], dim=1).detach().cpu().numpy()
        )
        return X_syn, y_syn

    def _inference_checkpoint_config(self) -> dict[str, Any]:
        """Serialize reference-neighbour interpolation settings."""
        return {
            "n_reference_neighbors": self.n_reference_neighbors,
            "observation_masking": self.observation_masking,
        }

    def _uses_conditional_masking(self) -> bool:
        """Use explicit opt-in observation masking for generation."""
        return self.observation_masking


class TabFORGEClassifier(TabFORGE, ClassifierMixin):
    """Classify rows by sampling a masked target token through diffusion."""

    _estimator_type_name = "classifier"

    def __init__(
        self,
        *,
        categorical_features: Sequence[int | str] | None = None,
        embedding_config: TabFORGEEmbeddingConfig | Mapping[str, Any] | None = None,
        architecture_config: TabFORGEArchitectureConfig | Mapping[str, Any] | None = None,
        diffusion_config: TabFORGEDiffusionConfig | Mapping[str, Any] | None = None,
        training_config: TabFORGETrainingConfig | Mapping[str, Any] | None = None,
        runtime_config: TabFORGERuntimeConfig | Mapping[str, Any] | None = None,
        random_state: int | None = None,
        logger: Any = None,
        n_prediction_samples: int = 32,
    ) -> None:
        """Configure diffusion-based target-token classification.

        Args:
            categorical_features: Raw categorical column names or positions.
            embedding_config: Embedding configuration object or field mapping.
            architecture_config: Neural architecture configuration.
            diffusion_config: EDM training and sampling configuration.
            training_config: Optimizer and validation configuration.
            runtime_config: Device, distribution, and logging configuration.
            random_state: Seed used for fitting and default inference.
            logger: Existing W&B run or compatible logger to reuse.
            n_prediction_samples: Target trajectories averaged per prediction.
        """
        super().__init__(
            categorical_features=categorical_features,
            embedding_config=embedding_config,
            architecture_config=architecture_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            runtime_config=runtime_config,
            random_state=random_state,
            logger=logger,
        )
        self.n_prediction_samples = n_prediction_samples

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any],
        *,
        validation_data: Any = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> "TabFORGEClassifier":
        """Fit the classifier from a checkpoint source or from scratch.

        Args:
            X: Raw training features.
            y: One-dimensional class labels.
            validation_data: Optional ``(X_valid, y_valid)`` pair.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-head loading policy.

        Returns:
            TabFORGEClassifier: The fitted classifier.
        """
        if (
            isinstance(self.n_prediction_samples, bool)
            or not isinstance(self.n_prediction_samples, (int, np.integer))
            or self.n_prediction_samples < 1
        ):
            raise ValueError("n_prediction_samples must be positive")
        return self._fit_with_task(
            X,
            y,
            "classification",
            validation_data=validation_data,
            checkpoint=checkpoint,
            detokeniser=detokeniser,
        )

    def predict_proba(self, X: pd.DataFrame | np.ndarray, *, random_state: int | None = None) -> np.ndarray:
        """Predict class probabilities from sampled target tokens.

        Args:
            X: Query features in the fitted raw schema.
            random_state: Optional seed overriding the estimator seed.

        Returns:
            np.ndarray: Probability matrix in ``classes_`` order.
        """
        if distributed_active():
            indices = shard_indices(len(X))
            local_X = _slice_rows(X, indices)
            seed = None if random_state is None else int(random_state) + distributed_rank()
            local = (
                self._predict_proba_local(local_X, seed)
                if len(indices)
                else np.empty((0, len(self.classes_)), dtype=np.float64)
            )
            return _gather_indexed(local, indices, len(X))
        return self._predict_proba_local(X, random_state)

    def _predict_proba_local(self, X: Any, random_state: int | None) -> np.ndarray:
        """Estimate probabilities for rows assigned to this rank.

        Args:
            X: Rank-local query rows.
            random_state: Rank-local sampling seed.
        """
        self._check_fitted()
        normalized = self._conditional_feature_latents(X)
        batch_size = self._prediction_batch_size(self.n_prediction_samples)
        probabilities = []
        for batch_index, start in enumerate(range(0, len(X), batch_size)):
            stop = min(start + batch_size, len(X))
            seed = random_state if batch_index == 0 else self._seed(random_state) + batch_index
            samples = self._predict_target_samples_from_features(normalized[start:stop], random_state=seed)
            probabilities.append(samples.reshape(stop - start, self.n_prediction_samples, -1).mean(axis=1))
        return np.concatenate(probabilities, axis=0)

    def predict(self, X: pd.DataFrame | np.ndarray, *, random_state: int | None = None) -> np.ndarray:
        """Predict the most probable class for each row.

        Args:
            X: Query features in the fitted raw schema.
            random_state: Optional seed overriding the estimator seed.
        """
        return self.classes_[self.predict_proba(X, random_state=random_state).argmax(axis=1)]

    def _predict_target_samples(self, X: Any, *, random_state: int | None) -> np.ndarray:
        """Sample decoded classification targets while clamping feature tokens.

        Args:
            X: Query features in the fitted raw schema.
            random_state: Seed for target-token trajectories.
        """
        self._check_fitted()
        if self.task_ != "classification":
            raise RuntimeError("The fitted estimator is not a classifier")
        normalized = self._conditional_feature_latents(X)
        return self._predict_target_samples_from_features(normalized, random_state=random_state)

    def _predict_target_samples_from_features(
        self,
        normalized: np.ndarray,
        *,
        random_state: int | None,
    ) -> np.ndarray:
        """Sample classification targets from precomputed feature latents.

        Args:
            normalized: Query latent grid normalized with fitted statistics.
            random_state: Seed for target-token trajectories.
        """
        conditional = self._prepare_conditional_latents_from_features(
            normalized, self.n_prediction_samples, random_state
        )
        latents = self._sample_latents(
            conditional["initial"],
            observation=conditional["observation"],
            observation_mask=conditional["observation_mask"],
            random_state=random_state,
        )
        target = self._decode_prediction_target(latents)
        return target.detach().cpu().numpy()

    def _inference_checkpoint_config(self) -> dict[str, Any]:
        """Serialize classifier trajectory settings."""
        return {
            "n_prediction_samples": self.n_prediction_samples,
        }


class TabFORGERegressor(TabFORGE, RegressorMixin):
    """Regress a scalar by sampling a masked target token through diffusion."""

    _estimator_type_name = "regressor"

    def __init__(
        self,
        *,
        categorical_features: Sequence[int | str] | None = None,
        embedding_config: TabFORGEEmbeddingConfig | Mapping[str, Any] | None = None,
        architecture_config: TabFORGEArchitectureConfig | Mapping[str, Any] | None = None,
        diffusion_config: TabFORGEDiffusionConfig | Mapping[str, Any] | None = None,
        training_config: TabFORGETrainingConfig | Mapping[str, Any] | None = None,
        runtime_config: TabFORGERuntimeConfig | Mapping[str, Any] | None = None,
        random_state: int | None = None,
        logger: Any = None,
        n_prediction_samples: int = 32,
    ) -> None:
        """Configure diffusion-based target regression.

        Args:
            categorical_features: Raw categorical column names or positions.
            embedding_config: Embedding configuration object or field mapping.
            architecture_config: Neural architecture configuration.
            diffusion_config: EDM training and sampling configuration.
            training_config: Optimizer and validation configuration.
            runtime_config: Device, distribution, and logging configuration.
            random_state: Seed used for fitting and default inference.
            logger: Existing W&B run or compatible logger to reuse.
            n_prediction_samples: Target trajectories retained per row.
        """
        super().__init__(
            categorical_features=categorical_features,
            embedding_config=embedding_config,
            architecture_config=architecture_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            runtime_config=runtime_config,
            random_state=random_state,
            logger=logger,
        )
        self.n_prediction_samples = n_prediction_samples

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any],
        *,
        validation_data: Any = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> "TabFORGERegressor":
        """Fit the regressor from a checkpoint source or from scratch.

        Args:
            X: Raw training features.
            y: One-dimensional continuous target values.
            validation_data: Optional ``(X_valid, y_valid)`` pair.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-head loading policy.

        Returns:
            TabFORGERegressor: The fitted regressor.
        """
        if (
            isinstance(self.n_prediction_samples, bool)
            or not isinstance(self.n_prediction_samples, (int, np.integer))
            or self.n_prediction_samples < 1
        ):
            raise ValueError("n_prediction_samples must be positive")
        return self._fit_with_task(
            X,
            y,
            "regression",
            validation_data=validation_data,
            checkpoint=checkpoint,
            detokeniser=detokeniser,
        )

    def predict_distribution(self, X: pd.DataFrame | np.ndarray, *, random_state: int | None = None) -> np.ndarray:
        """Return decoded target trajectories for each row.

        Args:
            X: Query features in the fitted raw schema.
            random_state: Optional seed overriding the estimator seed.

        Returns:
            np.ndarray: Array shaped ``(rows, n_prediction_samples)``.
        """
        if distributed_active():
            indices = shard_indices(len(X))
            local_X = _slice_rows(X, indices)
            seed = None if random_state is None else int(random_state) + distributed_rank()
            local = (
                self._predict_distribution_local(local_X, seed)
                if len(indices)
                else np.empty((0, self.n_prediction_samples), dtype=np.float64)
            )
            return _gather_indexed(local, indices, len(X))
        return self._predict_distribution_local(X, random_state)

    def _predict_distribution_local(self, X: Any, random_state: int | None) -> np.ndarray:
        """Return decoded trajectories for rows assigned to this rank.

        Args:
            X: Rank-local query rows.
            random_state: Rank-local sampling seed.
        """
        self._check_fitted()
        normalized = self._conditional_feature_latents(X)
        batch_size = self._prediction_batch_size(self.n_prediction_samples)
        distributions = []
        for batch_index, start in enumerate(range(0, len(X), batch_size)):
            stop = min(start + batch_size, len(X))
            seed = random_state if batch_index == 0 else self._seed(random_state) + batch_index
            distributions.append(self._predict_distribution_batch(normalized[start:stop], seed))
        return np.concatenate(distributions, axis=0)

    def _predict_distribution_batch(self, normalized: np.ndarray, random_state: int | None) -> np.ndarray:
        """Decode one memory-bounded batch of regression trajectories.

        Args:
            normalized: Precomputed normalized feature latents.
            random_state: Seed for target-token trajectories.
        """
        conditional = self._prepare_conditional_latents_from_features(
            normalized, self.n_prediction_samples, random_state
        )
        latents = self._sample_latents(
            conditional["initial"],
            observation=conditional["observation"],
            observation_mask=conditional["observation_mask"],
            random_state=random_state,
        )
        target = self._decode_prediction_target(latents)
        decoded = np.asarray(self.feature_processor_.inverse_target(target.detach().cpu().numpy()))
        return decoded.reshape(len(normalized), self.n_prediction_samples)

    def predict(self, X: pd.DataFrame | np.ndarray, *, random_state: int | None = None) -> np.ndarray:
        """Predict the mean decoded target for each row.

        Args:
            X: Query features in the fitted raw schema.
            random_state: Optional seed overriding the estimator seed.
        """
        self._check_fitted()
        return self.predict_distribution(X, random_state=random_state).mean(axis=1)

    def predict_std(self, X: pd.DataFrame | np.ndarray, *, random_state: int | None = None) -> np.ndarray:
        """Return trajectory-based predictive standard deviations.

        Args:
            X: Query features in the fitted raw schema.
            random_state: Optional seed overriding the estimator seed.
        """
        return self.predict_distribution(X, random_state=random_state).std(axis=1)

    def predict_interval(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        alpha: float = 0.05,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Return central trajectory-based prediction intervals.

        Args:
            X: Query features in the fitted raw schema.
            alpha: Probability mass excluded across both interval tails.
            random_state: Optional seed overriding the estimator seed.

        Returns:
            np.ndarray: Lower and upper bounds shaped ``(rows, 2)``.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        distribution = self.predict_distribution(X, random_state=random_state)
        return np.quantile(distribution, [alpha / 2, 1 - alpha / 2], axis=1).T

    def _inference_checkpoint_config(self) -> dict[str, Any]:
        """Serialize regressor trajectory settings."""
        return {
            "n_prediction_samples": self.n_prediction_samples,
        }


class _RepresentationMixin:
    """Share encoder/decoder representation extraction and token pooling."""

    def _extract_representation(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        source: str,
        layer: int | str | None,
        pooling: str | None,
    ) -> np.ndarray:
        """Extract and optionally pool a fitted TabFORGE representation.

        Args:
            X: Query features in the fitted raw schema.
            source: ``"encoder"`` or ``"decoder"``.
            layer: Zero-based layer or the relative encoder policy ``"middle"``
                or ``"final"``.
            pooling: ``None``, ``"mean"``, ``"max"``, ``"flatten"``, or
                supervised ``"label"`` pooling.
        """
        self._check_fitted()
        self._validate_representation_request(source, layer, pooling)
        resolved_layer = self._resolve_representation_layer(source, layer)
        self._record_representation_metadata(source, layer, pooling, resolved_layer)
        if distributed_active():
            indices = shard_indices(len(X))
            local = self._extract_representation_local(_slice_rows(X, indices), source=source, layer=resolved_layer)
            values = _gather_indexed(local, indices, len(X))
        else:
            values = self._extract_representation_local(X, source=source, layer=resolved_layer)
        return self._apply_pooling(values, pooling)

    def _extract_representation_local(self, X: Any, *, source: str, layer: int) -> np.ndarray:
        """Extract one rank's token representation.

        Args:
            X: Rank-local query rows.
            source: Validated representation source.
            layer: Validated optional layer index.
        """
        if source == "encoder":
            values = self.embedding_provider_.transform(X, processor=self.feature_processor_, layer=layer)
            return self._align_embedding_tokens(values)
        return self._extract_decoder_representation(X, layer=layer)

    def _extract_decoder_representation(self, X: Any, *, layer: int) -> np.ndarray:
        """Extract decoder hidden tokens without changing decoder outputs.

        Args:
            X: Rank-local query rows.
            layer: Zero-based decoder layer.
        """
        raw = self._full_embeddings_local(X)
        normalized = (raw - self.embedding_mean_.numpy()) / self.embedding_std_.numpy()
        latent = torch.as_tensor(normalized, dtype=torch.float32, device=self._device_)
        recovered = latent * self.embedding_std_.to(latent) + self.embedding_mean_.to(latent)
        with torch.no_grad():
            hidden = self.decoder_.hidden_representation(recovered, layer=layer)
        return np.ascontiguousarray(hidden.detach().cpu().numpy(), dtype=np.float32)

    @staticmethod
    def _validate_representation_request(source: str, layer: int | str | None, pooling: str | None) -> None:
        """Validate call-time representation selection.

        Args:
            source: Requested representation source.
            layer: Source-layer index or relative encoder policy.
            pooling: Optional token pooling operation.
        """
        if source not in {"encoder", "decoder"}:
            raise ValueError("source must be 'encoder' or 'decoder'")
        if isinstance(layer, str):
            if layer not in {"middle", "final"}:
                raise ValueError("layer must be an integer, None, 'middle', or 'final'")
        elif layer is not None and (isinstance(layer, bool) or not isinstance(layer, int) or layer < 0):
            raise ValueError("layer must be a non-negative integer, None, 'middle', or 'final'")
        if pooling not in {None, "mean", "max", "flatten", "label"}:
            raise ValueError("pooling must be None, 'mean', 'max', 'flatten', or 'label'")

    def _representation_layer_count(self, source: str) -> int:
        """Return the fitted physical layer count for one representation source."""
        if source == "encoder":
            return self.embedding_provider_.encoder_layer_count()
        if source == "decoder":
            return len(self.decoder_.transformer.encoder.layers)
        raise ValueError("source must be 'encoder' or 'decoder'")

    def _resolve_representation_layer(self, source: str, layer: int | str | None) -> int:
        """Resolve an optional request to one physical source-layer index."""
        count = self._representation_layer_count(source)
        configured = self.embedding_provider_.config.layer if source == "encoder" else None
        if isinstance(layer, str):
            if source != "encoder":
                raise ValueError("relative layer policies apply only to the encoder")
            resolved = count // 2 if layer == "middle" else count - 1
        else:
            resolved = configured if layer is None and configured is not None else layer
        if resolved is None:
            resolved = count - 1
        if resolved >= count:
            raise ValueError(f"{source} layer must be between 0 and {count - 1}")
        return int(resolved)

    def _record_representation_metadata(
        self,
        source: str,
        requested_layer: int | str | None,
        pooling: str | None,
        resolved_layer: int,
    ) -> None:
        """Record the last representation request and physical resolution."""
        self.representation_source_ = source
        self.representation_layer_ = requested_layer
        self.representation_pooling_ = pooling
        self.representation_layer_resolved_ = int(resolved_layer)
        self.representation_total_layers_ = self._representation_layer_count(source)
        self.representation_encoder_task_ = getattr(self.embedding_provider_, "encoder_task_", None)

    def _restore_representation_metadata(self, inference: Mapping[str, Any]) -> None:
        """Restore representation settings and fitted physical-layer metadata."""
        if not hasattr(self, "representation_source"):
            return
        source = inference.get("representation_source", self.representation_source)
        layer = inference.get("representation_layer", self.representation_layer)
        pooling = inference.get("representation_pooling", self.representation_pooling)
        resolved = inference.get("resolved_representation_layer")
        if resolved is None:
            resolved = self._resolve_representation_layer(source, layer)
        self._record_representation_metadata(source, layer, pooling, int(resolved))

    def _apply_pooling(self, values: np.ndarray, pooling: str | None) -> np.ndarray:
        """Apply a validated token pooling operation.

        Args:
            values: Three-dimensional token grid.
            pooling: Validated pooling operation.
        """
        if pooling is None:
            return values
        if pooling == "mean":
            return values.mean(axis=1)
        if pooling == "max":
            return values.max(axis=1)
        if pooling == "flatten":
            return values.reshape(len(values), -1)
        if self.task_ == "unsupervision":
            raise ValueError("label pooling requires a supervised estimator")
        position = self.feature_processor_.target_token_position_
        if position >= values.shape[1]:
            raise EmbeddingCapabilityError("The fitted embedding does not contain its target token")
        return values[:, position, :]


class TabFORGEEmbedder(_RepresentationMixin, TransformerMixin, TabFORGE):
    """Sklearn transformer exposing the full token grid and pooling helpers."""

    _estimator_type_name = "embedder"

    def __init__(
        self,
        *,
        task: str,
        categorical_features: Sequence[int | str] | None = None,
        embedding_config: TabFORGEEmbeddingConfig | Mapping[str, Any] | None = None,
        architecture_config: TabFORGEArchitectureConfig | Mapping[str, Any] | None = None,
        diffusion_config: TabFORGEDiffusionConfig | Mapping[str, Any] | None = None,
        training_config: TabFORGETrainingConfig | Mapping[str, Any] | None = None,
        runtime_config: TabFORGERuntimeConfig | Mapping[str, Any] | None = None,
        random_state: int | None = None,
        logger: Any = None,
    ) -> None:
        """Configure full-grid embeddings for an explicit task.

        Args:
            task: ``"classification"``, ``"regression"``, or ``"unsupervision"``.
            categorical_features: Raw categorical column names or positions.
            embedding_config: Embedding configuration object or field mapping.
            architecture_config: Neural architecture configuration.
            diffusion_config: EDM training and sampling configuration.
            training_config: Optimizer and validation configuration.
            runtime_config: Device, distribution, and logging configuration.
            random_state: Seed used for fitting and default transforms.
            logger: Existing W&B run or compatible logger to reuse.
        """
        super().__init__(
            categorical_features=categorical_features,
            embedding_config=embedding_config,
            architecture_config=architecture_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            runtime_config=runtime_config,
            random_state=random_state,
            logger=logger,
        )
        self.task = task

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any] | None = None,
        *,
        validation_data: Any = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> "TabFORGEEmbedder":
        """Fit the embedder with optional explicit validation data.

        Args:
            X: Raw training features.
            y: Target values for supervised tasks.
            validation_data: Optional validation features or supervised pair.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-head loading policy.

        Returns:
            TabFORGEEmbedder: The fitted transformer.
        """
        return self._fit_with_task(
            X,
            y,
            self.task,
            validation_data=validation_data,
            checkpoint=checkpoint,
            detokeniser=detokeniser,
        )

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        source: str = "encoder",
        layer: int | str | None = None,
        pooling: str | None = None,
    ) -> np.ndarray:
        """Transform rows into selected token or row-level representations.

        Args:
            X: Query features in the fitted raw schema.
            source: ``"encoder"`` or ``"decoder"`` representation source.
            layer: Zero-based source layer, ``"middle"``, ``"final"``, or
                ``None`` for the fitted/default layer.
            pooling: ``None``, ``"mean"``, ``"max"``, ``"flatten"``, or
                supervised ``"label"``.

        Returns:
            np.ndarray: Token grid or pooled row matrix.
        """
        return self._extract_representation(X, source=source, layer=layer, pooling=pooling)

    def flatten_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Flatten token and embedding dimensions for each row.

        Args:
            embeddings: Three-dimensional full token grid.

        Returns:
            np.ndarray: Matrix shaped ``(rows, tokens * dimension)``.
        """
        values = np.asarray(embeddings)
        if values.ndim != 3:
            raise ValueError(f"Expected embeddings with shape (rows, tokens, dimension), got {values.shape}")
        if hasattr(self, "embedding_shape_") and tuple(values.shape[1:]) != tuple(self.embedding_shape_):
            raise ValueError(f"Expected fitted embedding shape {self.embedding_shape_}, got {values.shape[1:]}")
        return values.reshape(values.shape[0], -1)

    def pool_embeddings(self, embeddings: np.ndarray, *, method: str = "mean") -> np.ndarray:
        """Pool token embeddings by mean or maximum.

        Args:
            embeddings: Three-dimensional full token grid.
            method: ``"mean"`` or ``"max"`` token reduction.

        Returns:
            np.ndarray: Matrix shaped ``(rows, embedding_dimension)``.
        """
        values = np.asarray(embeddings)
        if values.ndim != 3:
            raise ValueError(f"Expected embeddings with shape (rows, tokens, dimension), got {values.shape}")
        if hasattr(self, "embedding_shape_") and tuple(values.shape[1:]) != tuple(self.embedding_shape_):
            raise ValueError(f"Expected fitted embedding shape {self.embedding_shape_}, got {values.shape[1:]}")
        if method == "mean":
            return values.mean(axis=1)
        if method == "max":
            return values.max(axis=1)
        raise ValueError("method must be 'mean' or 'max'")


class TabFORGEClusterer(_RepresentationMixin, ClusterMixin, TabFORGE):
    """Cluster rows with KMeans over fitted TabFORGE representations."""

    _estimator_type_name = "clusterer"

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        categorical_features: Sequence[int | str] | None = None,
        embedding_config: TabFORGEEmbeddingConfig | Mapping[str, Any] | None = None,
        architecture_config: TabFORGEArchitectureConfig | Mapping[str, Any] | None = None,
        diffusion_config: TabFORGEDiffusionConfig | Mapping[str, Any] | None = None,
        training_config: TabFORGETrainingConfig | Mapping[str, Any] | None = None,
        runtime_config: TabFORGERuntimeConfig | Mapping[str, Any] | None = None,
        random_state: int | None = None,
        logger: Any = None,
        representation_source: str = "encoder",
        representation_layer: int | str | None = None,
        representation_pooling: str = "mean",
    ) -> None:
        """Configure representation-based KMeans clustering.

        Args:
            n_clusters: Number of clusters to discover.
            categorical_features: Raw categorical column names or positions.
            embedding_config: Embedding configuration object or field mapping.
            architecture_config: Neural architecture configuration.
            diffusion_config: EDM training and sampling configuration.
            training_config: Optimizer and validation configuration.
            runtime_config: Device, distribution, and logging configuration.
            random_state: Seed used for TabFORGE fitting and KMeans.
            logger: Existing W&B run or compatible logger to reuse.
            representation_source: Encoder or decoder representation source.
            representation_layer: Physical layer or relative encoder policy.
            representation_pooling: Token pooling operation.
        """
        super().__init__(
            categorical_features=categorical_features,
            embedding_config=embedding_config,
            architecture_config=architecture_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            runtime_config=runtime_config,
            random_state=random_state,
            logger=logger,
        )
        self.n_clusters = n_clusters
        self._set_representation_configuration(representation_source, representation_layer, representation_pooling)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any] | None = None,
        *,
        validation_data: Any = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> "TabFORGEClusterer":
        """Fit TabFORGE and the KMeans clustering head.

        Args:
            X: Raw training features.
            y: Ignored sklearn-compatible target argument.
            validation_data: Optional raw validation features.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-head loading policy.
        """
        del y
        if (
            isinstance(self.n_clusters, bool)
            or not isinstance(self.n_clusters, (int, np.integer))
            or self.n_clusters < 1
        ):
            raise ValueError("n_clusters must be a positive integer")
        self._fit_with_task(
            X,
            None,
            "unsupervision",
            validation_data=validation_data,
            checkpoint=checkpoint,
            detokeniser=detokeniser,
        )
        representation = self._default_representation(X)
        self.clusterer_ = KMeans(
            n_clusters=int(self.n_clusters),
            random_state=self.random_state,
            n_init="auto",
        ).fit(representation)
        self.labels_ = self.clusterer_.labels_.copy()
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Assign query rows to fitted clusters.

        Args:
            X: Query features in the fitted raw schema.
        """
        self._check_fitted()
        return self.clusterer_.predict(self._default_representation(X))

    def fit_predict(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any] | None = None,
        *,
        validation_data: Any = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> np.ndarray:
        """Fit the estimator and return training cluster labels.

        Args:
            X: Raw training features.
            y: Ignored sklearn-compatible target argument.
            validation_data: Optional raw validation features.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-head loading policy.
        """
        self.fit(
            X,
            y,
            validation_data=validation_data,
            checkpoint=checkpoint,
            detokeniser=detokeniser,
        )
        return self.labels_.copy()

    def _default_representation(self, X: Any) -> np.ndarray:
        """Extract the default clustering representation."""
        return self._extract_representation(
            X,
            source=self.representation_source,
            layer=self.representation_layer,
            pooling=self.representation_pooling,
        )

    def _set_representation_configuration(
        self,
        source: str,
        layer: int | str | None,
        pooling: str,
    ) -> None:
        """Validate and store the public clustering representation settings."""
        self._validate_representation_request(source, layer, pooling)
        if pooling == "label":
            raise ValueError("label pooling requires a supervised estimator")
        self.representation_source = source
        self.representation_layer = layer
        self.representation_pooling = pooling

    def _inference_checkpoint_config(self) -> dict[str, Any]:
        """Serialize clustering-head configuration."""
        return {
            "n_clusters": self.n_clusters,
            "representation_source": self.representation_source,
            "representation_layer": self.representation_layer,
            "representation_pooling": self.representation_pooling,
            "resolved_representation_layer": getattr(self, "representation_layer_resolved_", None),
            "representation_total_layers": getattr(self, "representation_total_layers_", None),
            "representation_encoder_task": getattr(self, "representation_encoder_task_", None),
        }

    def _inference_checkpoint_state(self) -> dict[str, Any]:
        """Serialize the fitted KMeans head."""
        return {"clusterer": self.clusterer_}

    def _load_inference_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore the fitted KMeans head.

        Args:
            state: Serialized clustering inference state.
        """
        if "clusterer" not in state:
            raise ValueError("Clusterer checkpoint is missing its fitted KMeans head")
        self.clusterer_ = state["clusterer"]
        self.labels_ = self.clusterer_.labels_.copy()


class TabFORGEAnomalyDetector(_RepresentationMixin, OutlierMixin, TabFORGE):
    """Detect anomalies with Isolation Forest over TabFORGE representations."""

    _estimator_type_name = "anomaly_detector"

    def __init__(
        self,
        *,
        contamination: str | float = "auto",
        categorical_features: Sequence[int | str] | None = None,
        embedding_config: TabFORGEEmbeddingConfig | Mapping[str, Any] | None = None,
        architecture_config: TabFORGEArchitectureConfig | Mapping[str, Any] | None = None,
        diffusion_config: TabFORGEDiffusionConfig | Mapping[str, Any] | None = None,
        training_config: TabFORGETrainingConfig | Mapping[str, Any] | None = None,
        runtime_config: TabFORGERuntimeConfig | Mapping[str, Any] | None = None,
        random_state: int | None = None,
        logger: Any = None,
        representation_source: str = "encoder",
        representation_layer: int | str | None = None,
        representation_pooling: str = "mean",
    ) -> None:
        """Configure representation-based anomaly detection.

        Args:
            contamination: Expected anomaly proportion or ``"auto"``.
            categorical_features: Raw categorical column names or positions.
            embedding_config: Embedding configuration object or field mapping.
            architecture_config: Neural architecture configuration.
            diffusion_config: EDM training and sampling configuration.
            training_config: Optimizer and validation configuration.
            runtime_config: Device, distribution, and logging configuration.
            random_state: Seed used for TabFORGE fitting and Isolation Forest.
            logger: Existing W&B run or compatible logger to reuse.
            representation_source: Encoder or decoder representation source.
            representation_layer: Physical layer or relative encoder policy.
            representation_pooling: Token pooling operation.
        """
        super().__init__(
            categorical_features=categorical_features,
            embedding_config=embedding_config,
            architecture_config=architecture_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            runtime_config=runtime_config,
            random_state=random_state,
            logger=logger,
        )
        self.contamination = contamination
        self._set_representation_configuration(representation_source, representation_layer, representation_pooling)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any] | None = None,
        *,
        validation_data: Any = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> "TabFORGEAnomalyDetector":
        """Fit TabFORGE and the Isolation Forest detector.

        Args:
            X: Raw training features.
            y: Ignored sklearn-compatible target argument.
            validation_data: Optional raw validation features.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-head loading policy.
        """
        del y
        self._validate_contamination()
        self._fit_with_task(
            X,
            None,
            "unsupervision",
            validation_data=validation_data,
            checkpoint=checkpoint,
            detokeniser=detokeniser,
        )
        self.detector_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        ).fit(self._default_representation(X))
        return self

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return Isolation Forest normality scores for query rows.

        Args:
            X: Query features in the fitted raw schema.
        """
        self._check_fitted()
        return self.detector_.score_samples(self._default_representation(X))

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return threshold-relative normality scores for query rows.

        Args:
            X: Query features in the fitted raw schema.
        """
        self._check_fitted()
        return self.detector_.decision_function(self._default_representation(X))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return ``+1`` for inliers and ``-1`` for anomalies.

        Args:
            X: Query features in the fitted raw schema.
        """
        self._check_fitted()
        return self.detector_.predict(self._default_representation(X))

    def _default_representation(self, X: Any) -> np.ndarray:
        """Extract the default anomaly representation."""
        return self._extract_representation(
            X,
            source=self.representation_source,
            layer=self.representation_layer,
            pooling=self.representation_pooling,
        )

    def _set_representation_configuration(
        self,
        source: str,
        layer: int | str | None,
        pooling: str,
    ) -> None:
        """Validate and store the public anomaly representation settings."""
        self._validate_representation_request(source, layer, pooling)
        if pooling == "label":
            raise ValueError("label pooling requires a supervised estimator")
        self.representation_source = source
        self.representation_layer = layer
        self.representation_pooling = pooling

    def _validate_contamination(self) -> None:
        """Validate the public anomaly-proportion setting."""
        if self.contamination == "auto":
            return
        if (
            isinstance(self.contamination, bool)
            or not isinstance(self.contamination, (int, float, np.number))
            or not 0 < float(self.contamination) <= 0.5
        ):
            raise ValueError("contamination must be 'auto' or a number in (0, 0.5]")

    def _inference_checkpoint_config(self) -> dict[str, Any]:
        """Serialize anomaly-head configuration."""
        return {
            "contamination": self.contamination,
            "representation_source": self.representation_source,
            "representation_layer": self.representation_layer,
            "representation_pooling": self.representation_pooling,
            "resolved_representation_layer": getattr(self, "representation_layer_resolved_", None),
            "representation_total_layers": getattr(self, "representation_total_layers_", None),
            "representation_encoder_task": getattr(self, "representation_encoder_task_", None),
        }

    def _inference_checkpoint_state(self) -> dict[str, Any]:
        """Serialize the fitted Isolation Forest head."""
        return {"detector": self.detector_}

    def _load_inference_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore the fitted Isolation Forest head.

        Args:
            state: Serialized anomaly-detection inference state.
        """
        if "detector" not in state:
            raise ValueError("Anomaly-detector checkpoint is missing its fitted Isolation Forest head")
        self.detector_ = state["detector"]


class TabFORGEImputer(TransformerMixin, TabFORGE):
    """Masked feature-token diffusion imputer."""

    _estimator_type_name = "imputer"

    def __init__(
        self,
        *,
        categorical_features: Sequence[int | str] | None = None,
        embedding_config: TabFORGEEmbeddingConfig | Mapping[str, Any] | None = None,
        architecture_config: TabFORGEArchitectureConfig | Mapping[str, Any] | None = None,
        diffusion_config: TabFORGEDiffusionConfig | Mapping[str, Any] | None = None,
        training_config: TabFORGETrainingConfig | Mapping[str, Any] | None = None,
        runtime_config: TabFORGERuntimeConfig | Mapping[str, Any] | None = None,
        random_state: int | None = None,
        logger: Any = None,
        n_imputation_samples: int = 32,
    ) -> None:
        """Configure conditional feature imputation.

        Args:
            categorical_features: Raw categorical column names or positions.
            embedding_config: Embedding configuration object or field mapping.
            architecture_config: Neural architecture configuration.
            diffusion_config: EDM training and sampling configuration.
            training_config: Optimizer and validation configuration.
            runtime_config: Device, distribution, and logging configuration.
            random_state: Seed used for fitting and default transforms.
            logger: Existing W&B run or compatible logger to reuse.
            n_imputation_samples: Conditional trajectories aggregated per cell.
        """
        super().__init__(
            categorical_features=categorical_features,
            embedding_config=embedding_config,
            architecture_config=architecture_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            runtime_config=runtime_config,
            random_state=random_state,
            logger=logger,
        )
        self.n_imputation_samples = n_imputation_samples

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any] | None = None,
        *,
        validation_data: Any = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> "TabFORGEImputer":
        """Fit the imputer from a checkpoint source or from scratch.

        Args:
            X: Raw training features.
            y: Reserved sklearn-compatible argument; must be ``None``.
            validation_data: Optional raw validation features.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-head loading policy.

        Returns:
            TabFORGEImputer: The fitted transformer.
        """
        if y is not None:
            raise ValueError("TabFORGEImputer.fit does not accept y")
        if (
            isinstance(self.n_imputation_samples, bool)
            or not isinstance(self.n_imputation_samples, (int, np.integer))
            or self.n_imputation_samples < 1
        ):
            raise ValueError("n_imputation_samples must be positive")
        return self._fit_with_task(
            X,
            None,
            "unsupervision",
            validation_data=validation_data,
            checkpoint=checkpoint,
            detokeniser=detokeniser,
        )

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        mask: np.ndarray | pd.DataFrame | None = None,
        random_state: int | None = None,
    ) -> Any:
        """Impute requested cells while preserving observed values.

        Args:
            X: Raw table containing values to condition on.
            mask: Boolean raw-column mask where ``True`` means regenerate. When
                omitted, missing cells in ``X`` are regenerated.
            random_state: Optional seed overriding the estimator seed.

        Returns:
            Any: Imputed data in the same table-container form as ``X``.
        """
        self._check_fitted()
        if distributed_active():
            indices = shard_indices(len(X))
            local_X = _slice_rows(X, indices)
            local_mask = None if mask is None else _slice_rows(mask, indices)
            seed = None if random_state is None else int(random_state) + distributed_rank()
            local = self._transform_local(local_X, local_mask, seed)
            return _gather_tabular(local, indices, len(X))
        return self._transform_local(X, mask, random_state)

    def _transform_local(self, X: Any, mask: Any, random_state: int | None) -> Any:
        """Impute rows assigned to the current process.

        Args:
            X: Rank-local raw rows.
            mask: Optional rank-local public regeneration mask.
            random_state: Rank-local trajectory seed.
        """
        inferred = self.feature_processor_.missing_mask(X)
        if mask is None:
            missing = inferred
        else:
            missing = np.asarray(mask, dtype=bool)
            if missing.shape != inferred.shape:
                raise ValueError("mask must have the same shape as X")
        if not missing.any():
            return X.copy(deep=True) if isinstance(X, pd.DataFrame) else np.asarray(X).copy()
        return self._transform_conditional(X, missing, random_state)

    def _transform_conditional(self, X: Any, missing: np.ndarray, random_state: int | None) -> Any:
        """Run conditional trajectories with observed latent tokens clamped.

        Args:
            X: Rank-local raw rows.
            missing: Raw-column mask where ``True`` marks cells to regenerate.
            random_state: Base seed for imputation trajectories.
        """
        reconstruction_values = []
        for trajectory in range(self.n_imputation_samples):
            seed = self._seed(random_state) + trajectory
            initial, observation, observed_mask = self._prepare_imputation_latents(X, missing, seed)
            latents = self._sample_latents(
                initial,
                observation=observation,
                observation_mask=observed_mask,
                random_state=seed,
            )
            feature_values, _ = self._decode(latents)
            reconstruction_values.append(feature_values.detach().cpu().numpy())
        aggregated = self._aggregate(reconstruction_values)
        decoded = self.feature_processor_.inverse_transform(aggregated, template=X)
        return self.feature_processor_.restore_observed(X, decoded, missing)

    def _prepare_imputation_latents(
        self,
        X: Any,
        missing: np.ndarray,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare noisy latents and mark observed feature tokens for clamping.

        Args:
            X: Raw rows to embed.
            missing: Raw-column regeneration mask.
            seed: Device random-stream seed.

        Returns:
            tuple: Initial noisy latents, clean observations, and the internal
            observed-token mask.
        """
        raw = self._full_embeddings_local(X)
        normalized = (raw - self.embedding_mean_.numpy()) / self.embedding_std_.numpy()
        observation = torch.as_tensor(normalized, dtype=torch.float32, device=self._device_)
        # Public masks use raw columns while latent grids use numerical-first tokens.
        embedding_mask = self.feature_processor_.embedding_mask(~missing)
        mask = torch.as_tensor(embedding_mask, dtype=torch.bool, device=self._device_)
        generator = torch.Generator(device=self._device_)
        generator.manual_seed(seed)
        initial = (
            observation
            + torch.randn(observation.shape, generator=generator, device=self._device_)
            * self.diffusion_config_.sigma_init
        )
        return initial, observation, mask

    def _aggregate(self, outputs: list[np.ndarray]) -> np.ndarray:
        """Aggregate decoded reconstruction values across trajectories.

        Args:
            outputs: Numerical values and categorical probabilities from each
                independently decoded trajectory.
        """
        return np.stack(outputs).mean(axis=0)

    def _inference_checkpoint_config(self) -> dict[str, Any]:
        """Serialize imputer trajectory settings."""
        return {
            "n_imputation_samples": self.n_imputation_samples,
        }


def _slice_rows(values: Any, indices: np.ndarray) -> Any:
    """Select rows from pandas or array-like input.

    Args:
        values: Source table or array.
        indices: Integer row positions to select.
    """
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return np.asarray(values)[indices]


def _gather_indexed(value: np.ndarray, indices: np.ndarray, total: int) -> np.ndarray:
    """Gather rank-local arrays and restore source row order.

    Args:
        value: Current rank's result array.
        indices: Original row positions represented by ``value``.
        total: Total rows across ranks.
    """
    result = np.empty((total, *value.shape[1:]), dtype=value.dtype)
    for rank_indices, rank_value in gather_objects((indices, value)):
        result[rank_indices] = rank_value
    return result


def _gather_tabular(value: Any, indices: np.ndarray, total: int) -> Any:
    """Gather rank-local tables and restore the original row ordering.

    Args:
        value: Current rank's pandas or NumPy result.
        indices: Original row positions represented by ``value``.
        total: Total rows across ranks.
    """
    pieces = gather_objects((indices, value))
    if isinstance(value, pd.DataFrame):
        frames = []
        for rank_indices, rank_value in pieces:
            frame = rank_value.copy()
            frame.index = rank_indices
            frames.append(frame)
        return pd.concat(frames).sort_index().reset_index(drop=True)
    result = np.empty((total, *np.asarray(value).shape[1:]), dtype=np.asarray(value).dtype)
    for rank_indices, rank_value in pieces:
        result[rank_indices] = rank_value
    return result
