"""Validated immutable configuration objects for TabFORGE estimators."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Mapping, TypeVar

T = TypeVar("T", bound="_Config")


@dataclass(frozen=True)
class _Config:
    """Base class shared by the public immutable configuration dataclasses."""

    def __post_init__(self) -> None:
        """Validate the selected embedding backend and dimensions."""
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool):
                continue
            if item.name in {"model_path", "device", "strategy"}:
                continue
            if isinstance(value, (int, float)) and value != value:
                raise ValueError(f"{type(self).__name__}.{item.name} cannot be NaN")

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable embedding configuration."""
        return asdict(self)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Expose config fields for sklearn-style inspection.

        Args:
            deep: Accepted for sklearn compatibility; config fields are flat.
        """
        # Frozen config fields are flat; ``deep`` is required by sklearn's API.
        del deep
        return self.to_dict()

    def __sklearn_clone__(self) -> _Config:
        """Keep frozen config identity stable when sklearn clones an estimator."""
        return self

    def set_params(self, **params: Any) -> _Config:
        """Return a validated replacement for direct config use.

        Args:
            **params: Dataclass fields to replace.
        """
        allowed = {item.name for item in fields(self)}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(f"Invalid configuration parameters: {', '.join(unknown)}")
        return replace(self, **params)


@dataclass(frozen=True)
class TabFORGEEmbeddingConfig(_Config):
    """Configure the Structure-aware Feature Encoder backend.

    Args:
        model_path: ``"auto"`` for the pinned external checkpoint, ``"mock"``
            for deterministic lightweight embeddings, or a local checkpoint path.
        n_folds: Held-out folds used for leakage-free training embeddings. Use
            zero to embed all training rows in one fitted context.
        layer: Optional zero-based encoder layer to extract.
        batch_size: Optional query batch size for the external encoder.
        n_estimators: External encoder ensemble members to average.
    """

    model_path: str = "auto"
    n_folds: int = 10
    layer: int | None = None
    batch_size: int | None = None
    n_estimators: int = 1

    def __post_init__(self) -> None:
        """Validate embedding configuration values."""
        super().__post_init__()
        if isinstance(self.n_folds, bool) or not isinstance(self.n_folds, int):
            raise TypeError("n_folds must be an integer")
        if self.n_folds < 0 or self.n_folds == 1:
            raise ValueError("n_folds must be 0 or at least 2")
        if self.layer is not None and (isinstance(self.layer, bool) or not isinstance(self.layer, int)):
            raise TypeError("layer must be an integer or None")
        if self.layer is not None and self.layer < 0:
            raise ValueError("layer must be non-negative or None")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive or None")
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, int):
            raise TypeError("n_estimators must be an integer")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")


@dataclass(frozen=True)
class TabFORGEArchitectureConfig(_Config):
    """Configure decoder and denoiser Transformer architecture.

    Args:
        decoder_layers: Transformer layers in the latent decoder.
        decoder_heads: Decoder attention heads.
        decoder_ffn_factor: Decoder feed-forward width factor.
        denoiser_layers: Transformer layers in the diffusion denoiser.
        denoiser_heads: Denoiser attention heads.
        denoiser_ffn_factor: Denoiser feed-forward width factor.
        embedding_dimension: Expected latent width. ``None`` infers it from the
            embedding provider.
    """

    decoder_layers: int = 4
    decoder_heads: int = 2
    decoder_ffn_factor: int = 10
    denoiser_layers: int = 4
    denoiser_heads: int = 4
    denoiser_ffn_factor: int = 16
    embedding_dimension: int | None = None

    def __post_init__(self) -> None:
        """Validate architecture configuration values."""
        super().__post_init__()
        for name in (
            "decoder_layers",
            "decoder_heads",
            "decoder_ffn_factor",
            "denoiser_layers",
            "denoiser_heads",
            "denoiser_ffn_factor",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dimension is not None and self.embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive or None")


@dataclass(frozen=True)
class TabFORGEDiffusionConfig(_Config):
    """Configure EDM training noise and reverse diffusion sampling.

    The sigma bounds, rho, log-normal sampling parameters, and data sigma follow
    the EDM parameterization. ``sigma_init`` sets the starting inference noise.
    """

    num_steps: int = 10
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_init: float = 0.1
    sigma_data: float = 1.0
    rho: float = 7.0
    p_mean: float = 0.0
    p_std: float = 1.0
    sigma_churn: float = 1.0
    enable_heun_correction: bool = False

    def __post_init__(self) -> None:
        """Validate diffusion configuration values."""
        super().__post_init__()
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if not 0 < self.sigma_min <= self.sigma_max:
            raise ValueError("sigma_min must be positive and no greater than sigma_max")
        if self.sigma_init <= 0 or self.sigma_data <= 0 or self.rho <= 0:
            raise ValueError("sigma_init, sigma_data, and rho must be positive")
        if self.p_std < 0 or self.sigma_churn < 0:
            raise ValueError("noise spread and churn cannot be negative")


@dataclass(frozen=True)
class TabFORGESchedulerConfig(_Config):
    """Configure one validation-driven scheduler in optimizer steps.

    Args:
        patience_steps: Updates without improvement before reducing the rate.
            ``None`` disables the scheduler for that phase.
        factor: Multiplicative learning-rate reduction.
        min_lr: Lower learning-rate bound.
        min_delta: Required validation-loss improvement.
    """

    patience_steps: int | None = 500
    factor: float = 0.5
    min_lr: float = 0.0
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        """Validate scheduler values."""
        super().__post_init__()
        if self.patience_steps is not None and (isinstance(self.patience_steps, bool) or self.patience_steps < 1):
            raise ValueError("patience_steps must be positive or None")
        if not 0 < self.factor < 1:
            raise ValueError("factor must be in (0, 1)")
        if self.min_lr < 0:
            raise ValueError("min_lr cannot be negative")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")


@dataclass(frozen=True)
class TabFORGETrainingConfig(_Config):
    """Configure phase budgets, optimizers, masking, and validation scheduling.

    Decoder and diffusion phases receive normalized shares of ``max_steps``.
    Conditional estimators sample enabled masking modes. Validation schedulers
    remain inactive when ``fit`` receives no validation split.
    """

    max_steps: int = 10_000
    batch_size: int = 512
    gradient_clip_value: float = 1.0
    decoder_phase_ratio: float = 0.5
    diffusion_phase_ratio: float = 0.5
    decoder_optimizer: str = "sgd"
    decoder_lr: float = 0.1
    decoder_weight_decay: float = 5e-5
    diffusion_optimizer: str = "adamw"
    diffusion_lr: float = 1e-3
    diffusion_weight_decay: float = 1e-5
    unmasked_probability: float = 0.50
    target_mask_probability: float = 0.25
    random_feature_mask_probability: float = 0.25
    target_loss_weight: float = 1.0
    # Scheduling remains dormant unless fit receives explicit validation data.
    decoder_scheduler: TabFORGESchedulerConfig | Mapping[str, Any] = field(default_factory=TabFORGESchedulerConfig)
    diffusion_scheduler: TabFORGESchedulerConfig | Mapping[str, Any] = field(
        default_factory=lambda: TabFORGESchedulerConfig(patience_steps=1000)
    )
    validation_every_n_epochs: int = 5

    def __post_init__(self) -> None:
        """Validate training configuration values by concern."""
        super().__post_init__()
        for name in ("decoder_scheduler", "diffusion_scheduler"):
            scheduler = getattr(self, name)
            if isinstance(scheduler, Mapping):
                scheduler = TabFORGESchedulerConfig(**dict(scheduler))
                object.__setattr__(self, name, scheduler)
            elif not isinstance(scheduler, TabFORGESchedulerConfig):
                raise TypeError(f"{name} must be a TabFORGESchedulerConfig or mapping")
        self._validate_training_budget()
        self._validate_optimizer_values()
        self._validate_masking_probabilities()
        self._validate_scheduler_values()

    def _validate_training_budget(self) -> None:
        """Validate phase lengths and checkpoint cadence."""
        if self.max_steps < 1 or self.batch_size < 1:
            raise ValueError("max_steps and batch_size must be positive")
        if self.gradient_clip_value < 0:
            raise ValueError("gradient_clip_value cannot be negative")
        if not math.isfinite(self.target_loss_weight) or self.target_loss_weight < 0:
            raise ValueError("target_loss_weight must be finite and nonnegative")
        if self.decoder_phase_ratio < 0 or self.diffusion_phase_ratio < 0:
            raise ValueError("phase ratios cannot be negative")
        if self.decoder_phase_ratio + self.diffusion_phase_ratio <= 0:
            raise ValueError("at least one training phase must be enabled")

    def _validate_optimizer_values(self) -> None:
        """Validate optimizer and gradient settings."""
        for name in (
            "decoder_lr",
            "diffusion_lr",
            "decoder_weight_decay",
            "diffusion_weight_decay",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def _validate_masking_probabilities(self) -> None:
        """Validate and normalize conditional masking probabilities."""
        probabilities = (
            self.unmasked_probability,
            self.target_mask_probability,
            self.random_feature_mask_probability,
        )
        if any(value < 0 for value in probabilities):
            raise ValueError("masking probabilities cannot be negative")
        if sum(probabilities) <= 0:
            raise ValueError("at least one masking probability must be positive")
        if any(value > 1 for value in probabilities):
            raise ValueError("masking probabilities must be at most 1")
        if sum(probabilities) > 1.0 + 1e-8:
            raise ValueError("enabled masking probabilities must sum to at most 1")

    def _validate_scheduler_values(self) -> None:
        """Validate validation-driven scheduler settings."""
        if isinstance(self.validation_every_n_epochs, bool) or self.validation_every_n_epochs < 1:
            raise ValueError("validation_every_n_epochs must be positive")

    def normalized_masking_probabilities(self, *, has_target: bool) -> dict[str, float]:
        """Return enabled masking-mode probabilities normalized to one.

        Args:
            has_target: Include the target-only masking mode when available.
        """
        enabled = {
            "unmasked": self.unmasked_probability,
            "target": self.target_mask_probability if has_target else 0.0,
            "random": self.random_feature_mask_probability,
        }
        total = sum(enabled.values())
        if total == 0:
            enabled["unmasked"] = 1.0
            total = 1.0
        return {key: value / total for key, value in enabled.items()}


@dataclass(frozen=True)
class TabFORGERuntimeConfig(_Config):
    """Configure standalone execution and optional experiment tracing.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or a concrete CUDA device.
        strategy: Single-process selection or ``"ddp"`` under ``torchrun``.
        gradient_accumulation: Sampled batches per optimizer update.
        deterministic: Enable seeded deterministic PyTorch operations.
        log_wandb: Create a standalone Weights & Biases run during fitting.
        wandb_project: Project used by an owned W&B run.
        wandb_entity: Optional W&B entity.
        wandb_dir: Local directory for W&B files.
    """

    device: str = "auto"
    strategy: str = "auto"
    gradient_accumulation: int = 1
    deterministic: bool = False
    log_wandb: bool = False
    wandb_project: str = "tabforge"
    wandb_entity: str | None = None
    wandb_dir: str = "./logs/wandb"

    def __post_init__(self) -> None:
        """Validate runtime configuration values."""
        super().__post_init__()
        if self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be positive")
        if self.device != "auto" and self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or a CUDA device such as 'cuda:0'")
        if self.strategy not in {"auto", "single_device", "ddp"}:
            raise ValueError("strategy must be 'auto', 'single_device', or 'ddp'")


_CONFIG_TYPES: dict[str, type[_Config]] = {
    "embedding_config": TabFORGEEmbeddingConfig,
    "architecture_config": TabFORGEArchitectureConfig,
    "diffusion_config": TabFORGEDiffusionConfig,
    "training_config": TabFORGETrainingConfig,
    "runtime_config": TabFORGERuntimeConfig,
}


def config_from_dict(value: T | Mapping[str, Any] | None, config_type: type[T]) -> T:
    """Resolve a public config object from ``None``, a mapping, or its type.

    Args:
        value: User-supplied config value.
        config_type: Expected immutable config class.
    """
    if value is None:
        return config_type()  # type: ignore[call-arg,return-value]
    if isinstance(value, config_type):
        return value
    if isinstance(value, Mapping):
        allowed = {item.name for item in fields(config_type)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise TypeError(f"Unknown {config_type.__name__} fields: {', '.join(unknown)}")
        return config_type(**dict(value))  # type: ignore[call-arg,return-value]
    raise TypeError(f"Expected {config_type.__name__}, mapping, or None; got {type(value).__name__}")


def resolved_config_dict(configs: Mapping[str, _Config]) -> dict[str, Any]:
    """Serialize a named collection of resolved configuration objects.

    Args:
        configs: Mapping from public config names to resolved objects.
    """
    return {name: config.to_dict() for name, config in configs.items()}
