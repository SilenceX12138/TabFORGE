"""Shared fitting, inference, and checkpoint lifecycle for TabFORGE estimators.

The public estimator classes delegate their common workflows to :class:`TabFORGE`.
This module owns fitted preprocessing and embedding state, component construction,
checkpoint migration, device selection, and latent sampling. Estimator-specific
prediction or generation behaviour lives in :mod:`tabforge.api.estimators`.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from tabforge._metadata import MODEL_ID, __version__
from tabforge.checkpoints import (
    CHECKPOINT_FILENAME,
    build_checkpoint,
    detokeniser_compatibility_reason,
    load_checkpoint,
    official_checkpoint_path,
)
from tabforge.checkpoints import save_checkpoint as write_checkpoint
from tabforge.checkpoints import (
    schema_from_processor,
)
from tabforge.config import (
    TabFORGEArchitectureConfig,
    TabFORGEDiffusionConfig,
    TabFORGEEmbeddingConfig,
    TabFORGERuntimeConfig,
    TabFORGETrainingConfig,
    config_from_dict,
)
from tabforge.distributed import active as distributed_active
from tabforge.distributed import gather_objects
from tabforge.distributed import initialize as initialize_distributed
from tabforge.distributed import local_cuda_device
from tabforge.distributed import rank as distributed_rank
from tabforge.distributed import shard_indices
from tabforge.distributed import world_size as distributed_world_size
from tabforge.embeddings import EmbeddingCapabilityError, TabPFNEmbeddingProvider
from tabforge.feature_processing import TabularFeatureProcessor
from tabforge.models import (
    EDMPreconditioner,
    TabFORGEDecoder,
    VariableColumnDenoiser,
    euler_heun_sample,
    power_mean_schedule,
)
from tabforge.training import TabFORGETrainer, TabFORGETrainingState


class TabFORGE(BaseEstimator):
    """Internal shared fitted core for TabFORGE estimators."""

    _estimator_type_name = "base"

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
    ) -> None:
        """Configure a TabFORGE estimator without fitting data-dependent state.

        Args:
            categorical_features: Raw categorical column names or integer positions.
            embedding_config: Embedding configuration object or field mapping.
            architecture_config: Decoder and denoiser architecture configuration.
            diffusion_config: EDM training and sampling configuration.
            training_config: Optimizer, phase, masking, and validation configuration.
            runtime_config: Device, distribution, determinism, and logging options.
            random_state: Default estimator seed.
            logger: Existing W&B run or compatible logger to reuse.
        """
        self.categorical_features = categorical_features
        self.embedding_config = embedding_config
        self.architecture_config = architecture_config
        self.diffusion_config = diffusion_config
        self.training_config = training_config
        self.runtime_config = runtime_config
        self.random_state = random_state
        self.logger = logger

    def set_params(self, **params: Any) -> "TabFORGE":
        """Set direct or nested sklearn-style estimator parameters.

        Args:
            **params: Direct estimator values or ``config__field`` nested values.

        Returns:
            TabFORGE: This estimator with updated constructor parameters.
        """
        nested: dict[str, dict[str, Any]] = {}
        direct = {}
        config_names = {
            "embedding_config": TabFORGEEmbeddingConfig,
            "architecture_config": TabFORGEArchitectureConfig,
            "diffusion_config": TabFORGEDiffusionConfig,
            "training_config": TabFORGETrainingConfig,
            "runtime_config": TabFORGERuntimeConfig,
        }
        for key, value in params.items():
            if "__" in key:
                parent, child = key.split("__", 1)
                if parent not in config_names:
                    direct[key] = value
                else:
                    nested.setdefault(parent, {})[child] = value
            else:
                direct[key] = value
        for name, values in nested.items():
            current = getattr(self, name)
            config_type = config_names[name]
            current_config = config_from_dict(current, config_type)
            setattr(self, name, current_config.set_params(**values))
        if direct:
            super().set_params(**direct)
        return self

    def full_embeddings(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform rows into the fitted full token embedding grid.

        Args:
            X: Query rows matching the fitted raw feature schema.

        Returns:
            np.ndarray: Grid shaped ``(rows, tokens, embedding_dimension)``.
        """
        self._check_fitted()
        if distributed_active():
            indices = shard_indices(len(X))
            local = self._full_embeddings_local(_slice_rows(X, indices))
            return _gather_indexed_arrays({"indices": indices, "embeddings": local}, len(X))
        return self._full_embeddings_local(X)

    def _full_embeddings_local(self, X: Any) -> np.ndarray:
        """Transform the rows assigned to the current process.

        Args:
            X: Rank-local query rows matching the fitted schema.
        """
        result = self.embedding_provider_.transform(X, processor=self.feature_processor_)
        return self._align_embedding_tokens(result)

    def save_backbone_checkpoint(self, path: str | Path) -> Path:
        """Save reusable decoder/diffusion backbones in the canonical format.

        Args:
            path: Destination ``.pt`` file or directory.

        This checkpoint deliberately excludes fitted data, embedding statistics,
        reference banks, token identities, observation-mask parameters, and
        dataset-specific reconstruction or target heads. The current estimator
        contains the best component states because ``fit`` restores them before
        returning. Use :meth:`save_checkpoint` to save a fitted estimator for
        standalone inference.
        """

        self._check_fitted()
        payload = build_checkpoint(
            state_dict=self._backbone_state(),
            config=self._backbone_checkpoint_config(),
            model_version=MODEL_ID,
            source={"family": "tabforge", "kind": "backbone"},
        )
        return _write_rank_zero(payload, self._checkpoint_destination(path))

    def _backbone_state(self) -> dict[str, dict[str, torch.Tensor]]:
        """Collect reusable decoder and diffusion Transformer weights."""
        return {
            "decoder": {
                f"transformer.{name}": parameter.detach().cpu().contiguous()
                for name, parameter in self.decoder_.transformer.named_parameters()
            },
            "diffusion": {
                f"transformer.{name}": parameter.detach().cpu().contiguous()
                for name, parameter in self.denoiser_.transformer.named_parameters()
            },
            "detokeniser": {},
        }

    def _backbone_checkpoint_config(self) -> dict[str, Any]:
        """Build the data-free backbone checkpoint metadata."""
        architecture = self.architecture_config_.to_dict()
        architecture["embedding_dimension"] = int(self.embedding_dimension_)
        return {"architecture_config": architecture}

    def initialize_from_checkpoint(
        self,
        path: str | Path = "pretrained",
        *,
        detokeniser: str = "auto",
    ) -> "TabFORGE":
        """Reuse compatible components while retaining current fitted data.

        Args:
            path: ``"pretrained"`` or a canonical checkpoint path.
            detokeniser: Schema-head policy: ``"auto"``, ``"load"``, or
                ``"reinitialize"``.

        This method is used during ``fit``. It preserves the current
        processor or embedding context; exact fitted-model restoration is
        provided separately by :meth:`restore_from_checkpoint`.
        """
        if not hasattr(self, "decoder_") or not hasattr(self, "denoiser_"):
            raise RuntimeError("Model components must be built before loading a checkpoint")
        if detokeniser not in {"auto", "load", "reinitialize"}:
            raise ValueError("detokeniser must be 'auto', 'load', or 'reinitialize'")
        checkpoint_path = official_checkpoint_path() if path == "pretrained" else path
        payload = load_checkpoint(checkpoint_path)
        self._validate_checkpoint_configuration(payload)
        self._apply_checkpoint_components(payload, detokeniser=detokeniser)
        self.checkpoint_source_ = str(checkpoint_path)
        return self

    def _validate_checkpoint_configuration(self, payload: Mapping[str, Any]) -> None:
        """Validate architecture fields before loading reusable tensors.

        Args:
            payload: Loaded canonical checkpoint mapping.
        """
        config = payload.get("config", {})
        saved = config.get("architecture_config", {})
        if not saved:
            return
        current = self.architecture_config_.to_dict()
        current["embedding_dimension"] = int(self.embedding_dimension_)
        for key in (
            "decoder_layers",
            "decoder_heads",
            "decoder_ffn_factor",
            "denoiser_layers",
            "denoiser_heads",
            "denoiser_ffn_factor",
            "embedding_dimension",
        ):
            # Older fitted checkpoints stored the unresolved public value even
            # though their tensors already encode the resolved dimension.
            if key == "embedding_dimension" and saved.get(key) is None:
                continue
            if key in saved and saved[key] != current.get(key):
                raise ValueError(
                    f"Checkpoint architecture {key}={saved[key]} does not match current model value {current.get(key)}"
                )

    def _apply_checkpoint_components(self, payload: Mapping[str, Any], *, detokeniser: str) -> None:
        """Load reusable backbones and apply the detokeniser policy.

        Args:
            payload: Validated canonical checkpoint mapping.
            detokeniser: Requested schema-dependent head loading policy.
        """
        state = payload["state_dict"]
        diffusion_state = dict(state.get("diffusion", {}))
        decoder_state = dict(state.get("decoder", {}))
        detokeniser_state = dict(state.get("detokeniser", {}))

        if diffusion_state:
            self._load_transformer_component(self.denoiser_, diffusion_state, "diffusion")
        if decoder_state:
            self._load_transformer_component(self.decoder_, decoder_state, "decoder")

        reason = detokeniser_compatibility_reason(payload.get("schema"), self._current_checkpoint_schema())
        self.detokeniser_compatibility_ = reason
        should_load = bool(detokeniser_state) and detokeniser != "reinitialize"
        if should_load and reason is not None:
            if detokeniser == "load":
                raise ValueError(f"Checkpoint detokeniser is incompatible: {reason}")
            should_load = False
        if should_load:
            self._load_detokeniser_state(detokeniser_state)
            self.detokeniser_loaded_ = True
        else:
            self.detokeniser_loaded_ = False

    @staticmethod
    def _extract_detokeniser_state(
        state: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Select schema-dependent decoder tensors for fitted checkpoints.

        Args:
            state: Complete decoder state mapping.
        """

        return {
            key: value
            for key, value in state.items()
            if key == "numerical_weight"
            or key.startswith("categorical_heads.")
            or key.startswith("target_head.")
            or key.startswith("prediction_head.")
        }

    def _load_transformer_component(
        self, module: torch.nn.Module, state: Mapping[str, torch.Tensor], label: str
    ) -> None:
        """Load exact reusable Transformer tensors and matching conditioning state.

        Args:
            module: Current decoder or denoiser module.
            state: Serialized component tensor mapping.
            label: Component label used in compatibility errors.
        """
        transformer = module.transformer
        transformer_state = {
            key.removeprefix("transformer."): value
            for key, value in state.items()
            if key == "transformer" or key.startswith("transformer.")
        }
        expected = transformer.state_dict()
        missing = sorted(set(expected) - set(transformer_state))
        unexpected = sorted(set(transformer_state) - set(expected))
        mismatched = [
            key
            for key in set(expected).intersection(transformer_state)
            if expected[key].shape != transformer_state[key].shape
            or expected[key].dtype != transformer_state[key].dtype
        ]
        if missing or unexpected or mismatched:
            details = ", ".join(
                part
                for part in (
                    f"missing={missing[:3]}" if missing else "",
                    f"unexpected={unexpected[:3]}" if unexpected else "",
                    f"mismatched={mismatched[:3]}" if mismatched else "",
                )
                if part
            )
            raise ValueError(f"{label} Transformer checkpoint is incompatible ({details})")
        transformer.load_state_dict(transformer_state, strict=True)

        # Denoiser conditioning state is reusable when its token-dependent
        # shapes match. Decoder output heads stay under the detokeniser policy.
        optional = {
            key: value
            for key, value in state.items()
            if label == "diffusion"
            and key in module.state_dict()
            and not key.startswith("transformer.")
            and module.state_dict()[key].shape == value.shape
            and module.state_dict()[key].dtype == value.dtype
        }
        if optional:
            module.load_state_dict(optional, strict=False)

    def _load_detokeniser_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load schema-dependent output heads with strict tensor checks.

        Args:
            state: Selected decoder output-head tensors.
        """
        current = self.decoder_.state_dict()
        selected = {}
        for key, value in state.items():
            if key not in current:
                continue
            if current[key].shape != value.shape or current[key].dtype != value.dtype:
                raise ValueError(f"Detokeniser tensor {key!r} has incompatible shape or dtype")
            selected[key] = value
        missing = sorted(set(state) - set(selected))
        if missing:
            raise ValueError(f"Detokeniser checkpoint entries are not present in this model: {missing[:3]}")
        self.decoder_.load_state_dict(selected, strict=False)

    def _current_checkpoint_schema(self) -> dict[str, Any]:
        """Describe fitted fields that determine decoder compatibility."""

        return schema_from_processor(self.feature_processor_, self.task_)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        include_reference_embeddings: str | bool = "auto",
    ) -> Path:
        """Save the standalone v1 checkpoint envelope for exact restoration.

        Args:
            path: Destination ``.pt`` file or directory.
            include_reference_embeddings: ``"auto"`` follows estimator type;
                booleans explicitly include or exclude the generation bank.

        Returns:
            Path: Canonical checkpoint file path.
        """
        self._check_fitted()
        if include_reference_embeddings == "auto":
            include_bank = self._supports_generation_estimator()
        elif isinstance(include_reference_embeddings, bool):
            include_bank = include_reference_embeddings
        else:
            raise ValueError("include_reference_embeddings must be 'auto', True, or False")
        if include_bank and not self._supports_generation_estimator():
            raise ValueError("Reference embeddings can only be included for a generation-capable estimator")
        destination = self._checkpoint_destination(path)
        return self._write_canonical_checkpoint(destination, include_reference_embeddings=include_bank)

    @staticmethod
    def _checkpoint_destination(path: str | Path) -> Path:
        """Resolve a checkpoint file while allowing a destination directory.

        Args:
            path: Requested file or directory destination.
        """
        destination = Path(path)
        if destination.suffix != ".pt":
            destination.mkdir(parents=True, exist_ok=True)
            destination = destination / CHECKPOINT_FILENAME
        return destination

    def _write_canonical_checkpoint(self, path: Path, *, include_reference_embeddings: bool) -> Path:
        """Serialize current model, schema, preprocessing, and progress state.

        Args:
            path: Resolved canonical checkpoint file.
            include_reference_embeddings: Include the dataset-bearing generation bank.
        """
        decoder_state = {
            f"transformer.{name}": value.detach().cpu().contiguous()
            for name, value in self.decoder_.transformer.state_dict().items()
        }
        state = {
            "diffusion": {key: value.detach().cpu().contiguous() for key, value in self.denoiser_.state_dict().items()},
            "decoder": decoder_state,
            "detokeniser": {
                key: value.detach().cpu().contiguous()
                for key, value in self._extract_detokeniser_state(self.decoder_.state_dict()).items()
            },
        }
        preprocessing = {
            "processor": self.feature_processor_,
            "embedding_mean": self.embedding_mean_.detach().cpu(),
            "embedding_std": self.embedding_std_.detach().cpu(),
            "embedding_provider_context": self.embedding_provider_.context_dict(),
        }
        inference_state = self._inference_checkpoint_state()
        if inference_state:
            preprocessing["inference_state"] = inference_state
        if include_reference_embeddings:
            preprocessing["reference_embeddings"] = self.reference_embeddings_.detach().cpu()
        training = {
            "step": getattr(self.training_state_, "global_step", 0),
            "epoch": getattr(self.training_state_, "epoch", 0),
            "state": asdict(self.training_state_) if self.training_state_ is not None else {},
        }
        payload = build_checkpoint(
            state_dict=state,
            schema=self._current_checkpoint_schema(),
            preprocessing=preprocessing,
            config=self._checkpoint_config(),
            training=training,
            model_version=MODEL_ID,
            source={"family": "tabforge", "kind": "fitted"},
        )
        return _write_rank_zero(payload, path)

    @classmethod
    def restore_from_checkpoint(cls, path: str | Path, *, device: str = "auto") -> "TabFORGE":
        """Restore the exact fitted model represented by a canonical checkpoint.

        Args:
            path: Canonical fitted checkpoint file or containing directory.
            device: Device selector used for restored neural components.
        """
        payload = load_checkpoint(path)
        return cls._restore_from_checkpoint_payload(payload, device=device)

    @classmethod
    def _restore_from_checkpoint_payload(cls, payload: Mapping[str, Any], *, device: str = "auto") -> "TabFORGE":
        """Restore an estimator from an already loaded canonical payload.

        Args:
            payload: Validated fitted checkpoint mapping.
            device: Device selector used for restored neural components.
        """
        preprocessing = payload.get("preprocessing", {})
        processor = preprocessing.get("processor")
        config = payload.get("config", {})
        if processor is None or not config.get("estimator_type"):
            raise ValueError("Canonical checkpoint does not contain fitted preprocessing and estimator configuration")
        if cls is TabFORGE:
            estimator_cls = _estimator_class(config["estimator_type"])
        elif config.get("estimator_type") != cls._estimator_type_name:
            raise ValueError("The checkpoint was created by a different estimator type")
        else:
            estimator_cls = cls
        estimator = cls._construct_checkpoint_estimator(estimator_cls, processor, config)
        estimator._load_canonical_checkpoint(payload, processor, device=device)
        return estimator

    def _load_canonical_checkpoint(self, payload: Mapping[str, Any], processor: Any, *, device: str) -> None:
        """Restore canonical tensors and fitted data without refitting.

        Args:
            payload: Validated fitted checkpoint mapping.
            processor: Serialized fitted feature processor.
            device: Device selector used for restored neural components.
        """
        config = payload["config"]
        self._load_checkpoint_config(config, processor, device)
        preprocessing = payload.get("preprocessing", {})
        state = payload["state_dict"]
        decoder_state = dict(state.get("decoder", {}))
        diffusion_state = dict(state.get("diffusion", {}))
        if not decoder_state or not diffusion_state:
            raise ValueError("Canonical fitted checkpoint is missing decoder or diffusion state")
        detokeniser_state = dict(state.get("detokeniser", {}))
        if not detokeniser_state:
            raise ValueError("Canonical fitted checkpoint is missing detokeniser state")
        self._load_transformer_component(self.decoder_, decoder_state, "decoder")
        self._load_detokeniser_state(detokeniser_state)
        self.denoiser_.load_state_dict(diffusion_state, strict=True)
        self.embedding_mean_ = preprocessing["embedding_mean"]
        self.embedding_std_ = preprocessing["embedding_std"]
        provider_context = preprocessing.get("embedding_provider_context")
        if provider_context is None:
            raise ValueError("Canonical fitted checkpoint is missing embedding-provider context")
        self.embedding_provider_ = TabPFNEmbeddingProvider.from_context(provider_context, device=str(self._device_))
        restore_representation = getattr(self, "_restore_representation_metadata", None)
        if restore_representation is not None:
            restore_representation(config.get("inference", {}))
        reference_embeddings = preprocessing.get("reference_embeddings")
        self.reference_embeddings_ = reference_embeddings
        self.training_state_ = _training_state_from_payload(payload.get("training", {}))
        self.generation_capable_ = self.reference_embeddings_ is not None
        self._load_inference_checkpoint_state(preprocessing.get("inference_state", {}))
        self._finalize_restored_model()

    @staticmethod
    def _construct_checkpoint_estimator(
        estimator_cls: type["TabFORGE"],
        processor: Any,
        config: dict[str, Any],
    ) -> "TabFORGE":
        """Construct an unfitted estimator from serialized public config.

        Args:
            estimator_cls: Concrete estimator class named by the checkpoint.
            processor: Fitted processor supplying categorical schema positions.
            config: Serialized estimator and inference configuration.
        """
        training_config = _migrate_training_config(config["training_config"])
        return estimator_cls(
            categorical_features=tuple(processor.input_categorical_indices_),
            embedding_config=TabFORGEEmbeddingConfig(**config["embedding_config"]),
            architecture_config=TabFORGEArchitectureConfig(**config["architecture_config"]),
            diffusion_config=TabFORGEDiffusionConfig(**config["diffusion_config"]),
            training_config=TabFORGETrainingConfig(**training_config),
            runtime_config=TabFORGERuntimeConfig(**config["runtime_config"]),
            random_state=config.get("random_state"),
            **_inference_constructor_args(estimator_cls, config),
        )

    # ================================================================
    # =                                                              =
    # =                       Model fitting                          =
    # =                                                              =
    # ================================================================

    def _fit_with_task(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[Any] | None,
        task: str,
        *,
        validation_data: Any | None = None,
        checkpoint: str | Path | None = "pretrained",
        detokeniser: str = "auto",
    ) -> "TabFORGE":
        """Fit preprocessing, embeddings, and trainable components for a task.

        Args:
            X: Raw training feature table.
            y: Optional one-dimensional training target.
            task: Classification, regression, or unsupervision contract.
            validation_data: Optional validation table or supervised pair.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-dependent decoder-head loading policy.
        """
        if detokeniser not in {"auto", "load", "reinitialize"}:
            raise ValueError("detokeniser must be 'auto', 'load', or 'reinitialize'")
        # === Validate training and optional validation inputs ===
        self._validate_fit_input(X, y, task)
        validation_X, validation_y = self._unpack_validation_data(validation_data, task)
        if validation_X is not None:
            self._validate_fit_input(validation_X, validation_y, task)
        self._resolve_fit_configuration(task)
        embeddings = self._fit_embedding_context(X, y, task)
        clean_latents = self._normalize_training_embeddings(embeddings)
        validation = self._transform_validation_data(validation_X, validation_y)
        self._prepare_trainable_components(clean_latents, checkpoint=checkpoint, detokeniser=detokeniser)
        targets = self.feature_processor_.transform_joint(X, y)
        self._train_components(clean_latents, targets, validation)
        self._finalize_fit_metadata(task)
        return self

    def _resolve_fit_configuration(self, task: str) -> None:
        """Resolve public configuration before fitting owned state.

        Args:
            task: Validated estimator task name.
        """
        self.embedding_config_ = config_from_dict(self.embedding_config, TabFORGEEmbeddingConfig)
        self.architecture_config_ = config_from_dict(self.architecture_config, TabFORGEArchitectureConfig)
        self.diffusion_config_ = config_from_dict(self.diffusion_config, TabFORGEDiffusionConfig)
        self.training_config_ = config_from_dict(self.training_config, TabFORGETrainingConfig)
        self.runtime_config_ = config_from_dict(self.runtime_config, TabFORGERuntimeConfig)
        initialize_distributed(self.runtime_config_.device)
        self._device_ = self._resolve_device(self.runtime_config_)
        self._configure_randomness()
        self.task_ = task

    def _fit_embedding_context(self, X: Any, y: Any, task: str) -> np.ndarray:
        """Fit preprocessing and embedding context on training rows.

        Args:
            X: Raw training feature table.
            y: Optional training target.
            task: Validated estimator task name.
        """
        self.feature_processor_ = TabularFeatureProcessor(self.categorical_features).fit(X)
        if y is not None:
            target_task = "classification" if task == "classification" else "regression"
            regression_transform = "standard" if self._estimator_type_name == "regressor" else "quantile"
            self.feature_processor_.fit_target(
                y,
                task=target_task,
                regression_transform=regression_transform,
            )
        self.embedding_provider_ = TabPFNEmbeddingProvider(
            self.embedding_config_,
            random_state=self.random_state,
            categorical_features=tuple(self.feature_processor_.categorical_embedding_positions_),
            device=str(self._device_),
        )
        requested_dimension = self.architecture_config_.embedding_dimension
        if requested_dimension is not None:
            self.embedding_provider_._requested_dimension = requested_dimension
        provider_target = None if task == "unsupervision" else np.asarray(y)
        embeddings = self._fit_embeddings_once(
            X,
            provider_target,
            task,
        )
        return self._align_embedding_tokens(embeddings)

    def _fit_embeddings_once(self, X: Any, y: Any, task: str) -> np.ndarray:
        """Compute training embeddings locally or across active ranks.

        Args:
            X: Raw training feature table.
            y: Optional training target.
            task: Validated estimator task name.
        """
        if not distributed_active():
            return self.embedding_provider_.fit(X, y, task=task, processor=self.feature_processor_)
        local = self.embedding_provider_.fit_shard(
            X,
            y,
            task=task,
            processor=self.feature_processor_,
            rank=distributed_rank(),
            world_size=distributed_world_size(),
        )
        result = _gather_indexed_arrays(local, len(X))
        return self.embedding_provider_.finalize_fit(result)

    def _normalize_training_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit latent normalization statistics and normalize training embeddings.

        Args:
            embeddings: Training grid shaped ``(rows, tokens, dimension)``.
        """
        self.embedding_dimension_ = int(embeddings.shape[-1])
        requested_dimension = self.architecture_config_.embedding_dimension
        if requested_dimension is not None and self.embedding_dimension_ != requested_dimension:
            raise ValueError(
                f"embedding_dimension={requested_dimension} does not match provider output {self.embedding_dimension_}"
            )
        self.embedding_shape_ = tuple(embeddings.shape[1:])
        self.embedding_mean_ = torch.as_tensor(embeddings.mean(axis=0), dtype=torch.float32)
        std = embeddings.std(axis=0, ddof=1 if len(embeddings) > 1 else 0)
        self.embedding_std_ = torch.as_tensor(np.where(std < 1e-6, 1.0, std), dtype=torch.float32)
        return (embeddings - self.embedding_mean_.numpy()) / self.embedding_std_.numpy()

    def _transform_validation_data(self, X: Any, y: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Transform validation rows without fitting preprocessing state.

        Args:
            X: Optional raw validation feature table.
            y: Optional validation target.
        """
        if X is None:
            return None, None
        embeddings = self._transform_embeddings_once(X)
        embeddings = self._align_embedding_tokens(embeddings)
        latents = (embeddings - self.embedding_mean_.numpy()) / self.embedding_std_.numpy()
        return latents, self.feature_processor_.transform_joint(X, y)

    def _transform_embeddings_once(self, X: Any) -> np.ndarray:
        """Transform validation rows across active ranks and restore row order.

        Args:
            X: Raw validation feature table.
        """
        if not distributed_active():
            return self.embedding_provider_.transform(X, processor=self.feature_processor_)
        indices = shard_indices(len(X))
        local_X = _slice_rows(X, indices)
        embeddings = self.embedding_provider_.transform(local_X, processor=self.feature_processor_)
        return _gather_indexed_arrays({"indices": indices, "embeddings": embeddings}, len(X))

    def _prepare_trainable_components(
        self,
        clean_latents: np.ndarray,
        *,
        checkpoint: str | Path | None,
        detokeniser: str,
    ) -> None:
        """Build model components and the generator reference bank.

        Args:
            clean_latents: Normalized training latent grid.
            checkpoint: Pretrained selector, canonical path, or ``None``.
            detokeniser: Schema-dependent decoder-head loading policy.
        """
        # Only generators retain the potentially large training embedding bank.
        self.reference_embeddings_ = (
            torch.as_tensor(clean_latents, dtype=torch.float32) if self._supports_generation_estimator() else None
        )
        self._build_components()
        if checkpoint == "pretrained" and self.embedding_config_.model_path == "mock":
            # The explicit mock backend is used for small offline fixtures and
            # has no architecture-compatible official encoder/backbone pair.
            # Production defaults use model_path="auto" and load the official
            # TabFORGE weights below.
            checkpoint = None
        if checkpoint is not None:
            self.initialize_from_checkpoint(checkpoint, detokeniser=detokeniser)

    def _train_components(self, clean_latents: np.ndarray, targets: np.ndarray, validation: tuple) -> None:
        """Train decoder and diffusion components with optional validation.

        Args:
            clean_latents: Normalized training latent grid.
            targets: Dense reconstruction targets aligned with training rows.
            validation: Optional normalized validation latents and targets.
        """
        validation_latents, validation_targets = validation
        self._adapt_supervised_training_batch_size(clean_latents.shape[1])
        trainer = TabFORGETrainer(
            training_config=self.training_config_,
            diffusion_config=self.diffusion_config_,
            device=self._device_,
            random_state=self.random_state,
            runtime_config=self.runtime_config_,
            logger=self.logger,
        )
        self.training_state_ = trainer.fit(
            self.decoder_,
            self.denoiser_,
            clean_latents,
            targets,
            has_target=self.task_ != "unsupervision",
            target_kind=self._target_kind(),
            conditional_masking=self._uses_conditional_masking(),
            target_cardinality=getattr(self.feature_processor_, "target_cardinality_", 0),
            numerical_count=len(self.feature_processor_.numerical_indices_),
            categorical_cardinalities=tuple(self.feature_processor_.cardinalities_),
            latent_mean=self.embedding_mean_,
            latent_std=self.embedding_std_,
            validation_latents=validation_latents,
            validation_targets=validation_targets,
            target_prediction=self._estimator_type_name in {"classifier", "regressor"},
        )
        if self._estimator_type_name == "regressor":
            self._fit_regression_target_head(clean_latents, targets)
        elif self._estimator_type_name == "classifier":
            self._fit_classification_target_head(clean_latents, targets)

    def _fit_classification_target_head(
        self,
        clean_latents: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Fit the compact classifier head with internal early stopping.

        Args:
            clean_latents: Normalized training latents including target tokens.
            targets: Dense classification reconstruction targets.
        """
        from sklearn.neural_network import MLPClassifier

        labels = targets[:, -self.feature_processor_.target_cardinality_ :].argmax(axis=1)
        if len(labels) < 20:
            return
        class_counts = np.bincount(labels)
        validation_rows = int(np.ceil(0.1 * len(labels)))
        early_stopping = class_counts.min() >= 2 and validation_rows >= len(class_counts)
        model = MLPClassifier(
            hidden_layer_sizes=(64,),
            alpha=10.0,
            max_iter=500,
            early_stopping=early_stopping,
            random_state=0 if self.random_state is None else int(self.random_state),
        ).fit(clean_latents[:, -1, :], labels)
        head = self.decoder_.prediction_head
        if not isinstance(head, torch.nn.Sequential) or len(head) != 3:
            raise RuntimeError("The fitted classifier requires a two-layer prediction head")
        with torch.no_grad():
            head[0].weight.copy_(torch.as_tensor(model.coefs_[0].T, dtype=head[0].weight.dtype))
            head[0].bias.copy_(torch.as_tensor(model.intercepts_[0], dtype=head[0].bias.dtype))
            output_weight = torch.as_tensor(model.coefs_[1].T, dtype=head[2].weight.dtype)
            output_bias = torch.as_tensor(model.intercepts_[1], dtype=head[2].bias.dtype)
            if output_weight.shape[0] == 1 and head[2].out_features == 2:
                head[2].weight.zero_()
                head[2].bias.zero_()
                head[2].weight[1].copy_(output_weight[0])
                head[2].bias[1].copy_(output_bias[0])
            else:
                head[2].weight.copy_(output_weight)
                head[2].bias.copy_(output_bias)

    def _fit_regression_target_head(
        self,
        clean_latents: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Fit the scalar decoder with cross-validated ridge regularization.

        Args:
            clean_latents: Normalized training latents including target tokens.
            targets: Dense regression reconstruction targets.
        """
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(
            clean_latents[:, -1, :],
            targets[:, -1],
        )
        head = self.decoder_.target_head
        if not isinstance(head, torch.nn.Linear) or head.out_features != 1:
            raise RuntimeError("The fitted regressor requires a scalar linear target head")
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(model.coef_, dtype=head.weight.dtype).reshape_as(head.weight))
            head.bias.copy_(torch.as_tensor(model.intercept_, dtype=head.bias.dtype).reshape_as(head.bias))

    def _adapt_supervised_training_batch_size(self, token_count: int) -> None:
        """Keep wide supervised tables within a fixed per-step token budget.

        Args:
            token_count: Fitted feature and target tokens per row.
        """
        if self.task_ == "unsupervision":
            return
        batch_size = min(self.training_config_.batch_size, max(1, 131_072 // token_count))
        if batch_size != self.training_config_.batch_size:
            self.training_config_ = replace(self.training_config_, batch_size=batch_size)

    def _prediction_batch_size(self, trajectory_count: int) -> int:
        """Bound simultaneous supervised trajectories by their token count.

        Args:
            trajectory_count: Independent trajectories generated per query row.
        """
        token_count = self.n_features_in_ + 1
        return max(1, 131_072 // (token_count * trajectory_count))

    def _target_kind(self) -> str | None:
        """Return the fitted target type used by decoder losses."""
        if self.task_ == "unsupervision":
            return None
        return "classification" if self.feature_processor_.target_is_classification_ else "regression"

    def _finalize_fit_metadata(self, task: str) -> None:
        """Expose fitted schema and task metadata through the estimator API.

        Args:
            task: Completed fit task name.
        """
        self.n_features_in_ = self.feature_processor_.n_features_in_
        self.feature_names_in_ = self.feature_processor_.feature_names_in_
        self.feature_schema_ = self.feature_processor_.schema_dict()
        if task == "classification":
            self.classes_ = self.feature_processor_.classes_.copy()
        self.generation_capable_ = self._supports_generation_estimator()
        self._fitted_ = True

    def _build_components(self) -> None:
        """Build decoder, denoiser, and EDM wrapper in dependency order."""
        target_kind, target_cardinality = self._component_target_spec()
        self.decoder_ = self._build_decoder(target_kind, target_cardinality)
        self.denoiser_ = self._build_denoiser()
        self.edm_ = EDMPreconditioner(self.denoiser_, self.diffusion_config_.sigma_data)
        self.decoder_.to(self._device_)
        self.denoiser_.to(self._device_)
        self.edm_.to(self._device_)

    def _component_target_spec(self) -> tuple[str | None, int]:
        """Resolve decoder target-head metadata from fitted preprocessing."""
        if self.task_ == "unsupervision":
            return None, 0
        target_kind = "classification" if self.feature_processor_.target_is_classification_ else "regression"
        return target_kind, getattr(self.feature_processor_, "target_cardinality_", 0)

    def _build_decoder(self, target_kind: str | None, target_cardinality: int) -> TabFORGEDecoder:
        """Construct the schema-aware latent decoder.

        Args:
            target_kind: Classification, regression, or ``None``.
            target_cardinality: Number of fitted target classes.
        """
        processor = self.feature_processor_
        architecture = self.architecture_config_
        return TabFORGEDecoder(
            n_feature_tokens=processor.n_features_in_,
            embedding_dimension=self.embedding_dimension_,
            numerical_count=len(processor.numerical_indices_),
            categorical_cardinalities=list(processor.cardinalities_),
            numerical_token_positions=list(processor.numerical_embedding_positions_),
            categorical_token_positions=list(processor.categorical_embedding_positions_),
            target_kind=target_kind,
            target_cardinality=target_cardinality,
            prediction_head_hidden=64 if self._estimator_type_name == "classifier" else None,
            layers=architecture.decoder_layers,
            heads=architecture.decoder_heads,
            ffn_factor=architecture.decoder_ffn_factor,
        )

    def _build_denoiser(self) -> VariableColumnDenoiser:
        """Construct the variable-column diffusion denoiser."""
        architecture = self.architecture_config_
        n_tokens = self.feature_processor_.feature_token_count(include_target=self.task_ != "unsupervision")
        return VariableColumnDenoiser(
            n_tokens=n_tokens,
            embedding_dimension=self.embedding_dimension_,
            layers=architecture.denoiser_layers,
            heads=architecture.denoiser_heads,
            ffn_factor=architecture.denoiser_ffn_factor,
        )

    def _align_embedding_tokens(self, embeddings: np.ndarray) -> np.ndarray:
        """Align backend token grids with the task-specific decoder layout.

        Args:
            embeddings: Provider output with row, token, and dimension axes.
        """
        result = np.asarray(embeddings, dtype=np.float32)
        expected = self.feature_processor_.feature_token_count(include_target=self.task_ != "unsupervision")
        if result.ndim != 3 or result.shape[1] < self.feature_processor_.n_features_in_:
            raise EmbeddingCapabilityError(
                f"Expected an embedding grid with at least {self.feature_processor_.n_features_in_} tokens, got {result.shape}"
            )
        if result.shape[1] == expected:
            return result
        if result.shape[1] == self.feature_processor_.n_features_in_ and expected > result.shape[1]:
            padding = np.zeros((len(result), 1, result.shape[2]), dtype=result.dtype)
            return np.concatenate([result, padding], axis=1)
        if result.shape[1] > expected:
            return result[:, :expected, :]
        raise EmbeddingCapabilityError(f"Expected {expected} embedding tokens, got {result.shape[1]}")

    def _prepare_conditional_latents(
        self,
        X: Any,
        trajectory_count: int,
        random_state: int | None,
    ) -> dict[str, torch.Tensor]:
        """Prepare repeated feature latents with the target token unobserved.

        Args:
            X: Raw query features.
            trajectory_count: Target trajectories per query row.
            random_state: Sampling seed.
        """
        normalized = self._conditional_feature_latents(X)
        return self._prepare_conditional_latents_from_features(normalized, trajectory_count, random_state)

    def _conditional_feature_latents(self, X: Any) -> np.ndarray:
        """Embed and normalize conditional feature rows once.

        Args:
            X: Raw query features.
        """
        raw = self._full_embeddings_local(X)
        return (raw - self.embedding_mean_.numpy()) / self.embedding_std_.numpy()

    def _prepare_conditional_latents_from_features(
        self,
        normalized: np.ndarray,
        trajectory_count: int,
        random_state: int | None,
    ) -> dict[str, torch.Tensor]:
        """Repeat precomputed feature latents and initialize target trajectories.

        Args:
            normalized: Precomputed normalized feature latent grid.
            trajectory_count: Target trajectories per query row.
            random_state: Sampling seed.
        """
        normalized = np.repeat(normalized, trajectory_count, axis=0)
        observation = torch.as_tensor(normalized, dtype=torch.float32, device=self._device_)
        observation_mask = torch.ones(
            (len(normalized), observation.shape[1]),
            dtype=torch.bool,
            device=self._device_,
        )
        observation_mask[:, -1] = False
        generator = torch.Generator(device=self._device_)
        generator.manual_seed(self._seed(random_state))
        initial = (
            observation
            + torch.randn(observation.shape, generator=generator, device=self._device_)
            * self.diffusion_config_.sigma_init
        )
        return {
            "initial": initial,
            "observation": observation,
            "observation_mask": observation_mask,
        }

    def _sample_latents(
        self,
        initial: torch.Tensor,
        *,
        observation: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
        random_state: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample clean latent grids from the fitted diffusion model.

        Args:
            initial: Initial noisy latent grid.
            observation: Optional clean latent grid to clamp.
            observation_mask: Token mask where ``True`` means observed.
            random_state: Seed used when ``generator`` is absent.
            generator: Optional caller-owned device random stream.
        """
        sigmas = self._sampling_schedule(initial.device)
        if generator is None:
            generator = torch.Generator(device=initial.device)
            generator.manual_seed(self._seed(random_state))
        diffusion = self.diffusion_config_
        with torch.no_grad():
            self.edm_.eval()
            return euler_heun_sample(
                self.edm_,
                initial,
                sigmas,
                observation=observation,
                observation_mask=observation_mask,
                generator=generator,
                enable_heun_correction=diffusion.enable_heun_correction,
                sigma_churn=diffusion.sigma_churn,
            )

    def _sampling_schedule(self, device: torch.device) -> torch.Tensor:
        """Build the fitted diffusion schedule from its configured start noise.

        Args:
            device: Torch device that owns the sampling tensors.
        """
        diffusion = self.diffusion_config_
        sigma_start = min(max(diffusion.sigma_init, diffusion.sigma_min), diffusion.sigma_max)
        # The terminal zero performs the final clean-data transition.
        return power_mean_schedule(
            diffusion.num_steps,
            sigma_min=diffusion.sigma_min,
            sigma_max=sigma_start,
            rho=diffusion.rho,
            device=device,
        )

    def _decode(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode latent grids into feature and optional target reconstructions.

        Args:
            latents: Normalized clean latent grid.
        """
        # Diffusion operates in normalized space while the decoder uses raw embeddings.
        recovered = latents * self.embedding_std_.to(latents) + self.embedding_mean_.to(latents)
        output = self.decoder_(recovered)
        processor = self.feature_processor_
        values = latents.new_zeros((len(latents), processor.reconstruction_dim_))
        if output["numerical"].shape[1]:
            values[:, : len(processor.numerical_indices_)] = output["numerical"]
        offset = len(processor.numerical_indices_)
        for cardinality, logits in zip(processor.cardinalities_, output["categorical"]):
            values[:, offset : offset + cardinality] = logits.softmax(dim=1)
            offset += cardinality
        target = output["target"]
        if target is not None and self.task_ == "classification":
            target = target.softmax(dim=1)
        return values, target

    def _decode_prediction_target(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the target token through the lightweight predictor head.

        Args:
            latents: Normalized sampled latent grid containing a target token.
        """
        target = self.decoder_.predict_target(latents)
        return target.softmax(dim=1) if self.task_ == "classification" else target

    def _validate_fit_input(self, X: Any, y: Sequence[Any] | None, task: str) -> None:
        """Validate the core table and target requirements for fitting.

        Args:
            X: Raw feature table.
            y: Optional target values.
            task: Requested fit task.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or NumPy array")
        if len(X) < 1:
            raise ValueError("X must contain at least one row")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if task not in {"classification", "regression", "unsupervision"}:
            raise ValueError("task must be 'classification', 'regression', or 'unsupervision'")
        if task in {"classification", "regression"} and y is None:
            raise ValueError(f"{self.__class__.__name__}.fit requires y")
        if task == "unsupervision" and y is not None:
            raise ValueError(f"{self.__class__.__name__}.fit does not accept y for task='unsupervision'")

    def _unpack_validation_data(self, validation_data: Any | None, task: str) -> tuple[Any | None, Any | None]:
        """Normalize public validation input into feature and target values.

        Args:
            validation_data: Public validation table or supervised pair.
            task: Requested fit task.
        """
        if validation_data is None:
            return None, None
        if task == "unsupervision":
            if isinstance(validation_data, tuple):
                if len(validation_data) != 2 or validation_data[1] is not None:
                    raise ValueError("unsupervision validation_data must be X_valid or (X_valid, None)")
                return validation_data
            return validation_data, None
        if not isinstance(validation_data, tuple) or len(validation_data) != 2:
            raise ValueError("supervised validation_data must be a (X_valid, y_valid) tuple")
        validation_X, validation_y = validation_data
        if validation_y is None:
            raise ValueError("supervised validation_data requires y_valid")
        return validation_X, validation_y

    def _supports_generation_estimator(self) -> bool:
        """Return whether this estimator retains generation reference state."""
        return self._estimator_type_name == "generator"

    def _uses_conditional_masking(self) -> bool:
        """Return whether training samples observation masks for this task."""
        return not self._supports_generation_estimator()

    def _check_fitted(self) -> None:
        """Require a completed fit before inference or serialization."""
        if not getattr(self, "_fitted_", False):
            raise RuntimeError(f"{self.__class__.__name__} is not fitted")

    def _resolve_device(self, config: TabFORGERuntimeConfig) -> torch.device:
        """Resolve public runtime options to one concrete torch device.

        Args:
            config: Validated runtime configuration.
        """
        device = local_cuda_device(config.device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
        return resolved

    def _configure_randomness(self) -> None:
        """Seed owned initialization and honor the deterministic runtime flag."""
        if self.random_state is None and not self.runtime_config_.deterministic:
            return
        seed = 0 if self.random_state is None else int(self.random_state)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.runtime_config_.deterministic:
            if self._device_.type == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)

    def _seed(self, override: int | None) -> int:
        """Resolve an operation-specific or estimator-level random seed.

        Args:
            override: Optional operation-specific seed.
        """
        if override is not None:
            return int(override)
        if self.random_state is not None:
            return int(self.random_state)
        return int(np.random.SeedSequence().generate_state(1)[0])

    def _checkpoint_config(self) -> dict[str, Any]:
        """Build the complete configuration for exact checkpoint restoration."""
        architecture = self.architecture_config_.to_dict()
        architecture["embedding_dimension"] = int(self.embedding_dimension_)
        return {
            "model_id": MODEL_ID,
            "package_version": __version__,
            "random_state": self.random_state,
            "task": self.task_,
            "estimator_type": self._estimator_type_name,
            "embedding_config": self.embedding_config_.to_dict(),
            "architecture_config": architecture,
            "diffusion_config": self.diffusion_config_.to_dict(),
            "training_config": self.training_config_.to_dict(),
            "runtime_config": self.runtime_config_.to_dict(),
            "embedding_shape": list(self.embedding_shape_),
            "feature_schema": self.feature_processor_.schema_dict(),
            "inference": self._inference_checkpoint_config(),
        }

    def _inference_checkpoint_config(self) -> dict[str, Any]:
        """Return estimator-specific inference settings for serialization."""
        return {}

    def _inference_checkpoint_state(self) -> dict[str, Any]:
        """Return fitted estimator-specific inference state for serialization."""
        return {}

    def _load_inference_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore fitted estimator-specific inference state.

        Args:
            state: Serialized estimator-specific inference state.
        """
        del state

    def _load_checkpoint_config(self, config: dict[str, Any], processor: Any, device: str) -> None:
        """Restore fitted configuration and build empty model components.

        Args:
            config: Serialized resolved estimator configuration.
            processor: Serialized fitted feature processor.
            device: Requested restoration device.
        """
        initialize_distributed(device)
        self.task_ = config["task"]
        self.embedding_config_ = TabFORGEEmbeddingConfig(**config["embedding_config"])
        self.architecture_config_ = TabFORGEArchitectureConfig(**config["architecture_config"])
        self.diffusion_config_ = TabFORGEDiffusionConfig(**config["diffusion_config"])
        self.training_config_ = TabFORGETrainingConfig(**_migrate_training_config(config["training_config"]))
        self.runtime_config_ = TabFORGERuntimeConfig(**config["runtime_config"])
        self.feature_processor_ = processor
        self.embedding_shape_ = tuple(config["embedding_shape"])
        if len(self.embedding_shape_) != 2 or any(int(value) < 1 for value in self.embedding_shape_):
            raise ValueError(f"Invalid fitted embedding shape in checkpoint: {self.embedding_shape_}")
        self.embedding_dimension_ = int(self.embedding_shape_[-1])
        resolved_device = local_cuda_device(device)
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_ = torch.device(resolved_device)
        self._build_components()

    def _finalize_restored_model(self) -> None:
        """Expose restored schema and task metadata through the estimator API."""
        self.n_features_in_ = self.feature_processor_.n_features_in_
        self.feature_names_in_ = self.feature_processor_.feature_names_in_
        self.feature_schema_ = self.feature_processor_.schema_dict()
        if self.task_ == "classification":
            self.classes_ = self.feature_processor_.classes_.copy()
        self._fitted_ = True


def _training_state_from_payload(
    training: Mapping[str, Any],
) -> TabFORGETrainingState | None:
    """Recreate training progress metadata from a canonical checkpoint.

    Args:
        training: Serialized checkpoint training section.
    """
    state = training.get("state")
    if not state:
        return None
    return TabFORGETrainingState(**dict(state))


def _gather_indexed_arrays(local: Mapping[str, np.ndarray], total: int) -> np.ndarray:
    """Gather rank-local embedding arrays and restore source row order.

    Args:
        local: Current rank's row indices and embedding array.
        total: Total rows across ranks.
    """
    pieces = gather_objects(dict(local))
    sample = next(piece["embeddings"] for piece in pieces if len(piece["indices"]))
    result = np.empty((total, *sample.shape[1:]), dtype=sample.dtype)
    for piece in pieces:
        result[piece["indices"]] = piece["embeddings"]
    return result


def _slice_rows(values: Any, indices: np.ndarray) -> Any:
    """Select rows from pandas or array-like inputs.

    Args:
        values: Source table or array.
        indices: Integer row positions.
    """
    return values.iloc[indices] if hasattr(values, "iloc") else np.asarray(values)[indices]


def _migrate_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate shared legacy scheduler fields into phase configurations.

    Args:
        config: Serialized training configuration from a checkpoint.
    """
    result = dict(config)
    if "scheduler_patience" not in result:
        return result
    scheduler = {
        "patience_steps": result.pop("scheduler_patience"),
        "factor": result.pop("scheduler_factor", 0.5),
        "min_lr": result.pop("scheduler_min_lr", 0.0),
    }
    result.setdefault("decoder_scheduler", scheduler)
    result.setdefault("diffusion_scheduler", dict(scheduler))
    return result


def restore_from_checkpoint(path: str | Path, *, device: str = "auto") -> TabFORGE:
    """Restore the concrete estimator stored in a canonical checkpoint.

    Args:
        path: Canonical fitted checkpoint file or containing directory.
        device: Device selector for restored neural components.
    """
    return TabFORGE.restore_from_checkpoint(path, device=device)


def _estimator_class(name: str) -> type[TabFORGE]:
    """Resolve a serialized estimator type without import cycles.

    Args:
        name: Canonical estimator type identifier.
    """
    from .estimators import (
        TabFORGEAnomalyDetector,
        TabFORGEClassifier,
        TabFORGEClusterer,
        TabFORGEEmbedder,
        TabFORGEGenerator,
        TabFORGEImputer,
        TabFORGERegressor,
    )

    return {
        "generator": TabFORGEGenerator,
        "classifier": TabFORGEClassifier,
        "regressor": TabFORGERegressor,
        "embedder": TabFORGEEmbedder,
        "imputer": TabFORGEImputer,
        "clusterer": TabFORGEClusterer,
        "anomaly_detector": TabFORGEAnomalyDetector,
    }[name]


def _inference_constructor_args(estimator_cls: type, config: dict[str, Any]) -> dict[str, Any]:
    """Restore estimator-specific constructor options from a checkpoint.

    Args:
        estimator_cls: Concrete estimator class being restored.
        config: Serialized checkpoint configuration.
    """
    inference = config.get("inference", {})
    if estimator_cls.__name__ == "TabFORGEGenerator":
        return {
            "task": config["task"],
            "n_reference_neighbors": inference.get("n_reference_neighbors", 1),
            "observation_masking": inference.get("observation_masking", False),
        }
    if estimator_cls.__name__ == "TabFORGEClassifier":
        return {
            "n_prediction_samples": inference.get("n_prediction_samples", 32),
        }
    if estimator_cls.__name__ == "TabFORGERegressor":
        return {
            "n_prediction_samples": inference.get("n_prediction_samples", 32),
        }
    if estimator_cls.__name__ == "TabFORGEImputer":
        return {
            "n_imputation_samples": inference.get("n_imputation_samples", 32),
        }
    if estimator_cls.__name__ == "TabFORGEEmbedder":
        return {"task": config["task"]}
    if estimator_cls.__name__ == "TabFORGEClusterer":
        legacy = "representation_source" not in inference
        return {
            "n_clusters": inference.get("n_clusters", 8),
            "representation_source": inference.get("representation_source", "encoder"),
            "representation_layer": inference.get("representation_layer", 7 if legacy else None),
            "representation_pooling": inference.get("representation_pooling", "flatten" if legacy else "mean"),
        }
    if estimator_cls.__name__ == "TabFORGEAnomalyDetector":
        legacy = "representation_source" not in inference
        return {
            "contamination": inference.get("contamination", "auto"),
            "representation_source": inference.get("representation_source", "encoder"),
            "representation_layer": inference.get("representation_layer", 7 if legacy else None),
            "representation_pooling": inference.get("representation_pooling", "flatten" if legacy else "mean"),
        }
    return {}


def _write_rank_zero(payload: Mapping[str, Any], path: Path) -> Path:
    """Write a checkpoint only on rank zero during distributed execution.

    Args:
        payload: Canonical checkpoint mapping.
        path: Destination checkpoint file.
    """
    if not distributed_active() or torch.distributed.get_rank() == 0:
        result = write_checkpoint(payload, path)
    else:
        result = path
    return result
