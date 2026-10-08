"""Direct phase-based training loop for TabFORGE neural components.

The trainer keeps embedding arrays on CPU, shards rows for distributed execution,
and independently optimizes decoder and diffusion phases. Validation, scheduling,
best-state selection, progress rendering, and optional tracing are coordinated here.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from tabforge.config import (
    TabFORGEDiffusionConfig,
    TabFORGERuntimeConfig,
    TabFORGESchedulerConfig,
    TabFORGETrainingConfig,
)
from tabforge.distributed import active as distributed_active
from tabforge.models import (
    EDMPreconditioner,
    TabFORGEDecoder,
    VariableColumnDenoiser,
    edm_weighted_mse,
)
from tabforge.models.losses import mixed_feature_loss
from tabforge.training.logging import TrainingLogger

_PROGRESS_BAR_WIDTH = 24
_PROGRESS_REFRESH_COUNT = 100

# ================================================================
# =                                                              =
# =                Training state and interface                  =
# =                                                              =
# ================================================================


@dataclass
class TabFORGETrainingState:
    """Store completed training progress and selection losses."""

    global_step: int = 0
    epoch: int = 0
    phase: str = "diffusion"
    phase_step: int = 0
    decoder_optimizer_step: int = 0
    diffusion_optimizer_step: int = 0
    best_diffusion_loss: float | None = None
    best_decoder_loss: float | None = None


class TabFORGETrainer:
    """Direct trainer used by estimator methods and torch-distributed workers.

    When the caller has initialized a process group, the same phase loop uses
    synchronized gradients across workers. The ordinary estimator path stays
    direct and works without a process group.
    """

    def __init__(
        self,
        *,
        training_config: TabFORGETrainingConfig,
        diffusion_config: TabFORGEDiffusionConfig,
        device: torch.device,
        random_state: int | None,
        runtime_config: TabFORGERuntimeConfig | None = None,
        logger: Any = None,
    ) -> None:
        """Configure direct component training.

        Args:
            training_config: Resolved optimizer and phase configuration.
            diffusion_config: Resolved EDM training and sampling configuration.
            device: Torch device used by tensors and neural components.
            random_state: Optional seed for deterministic random streams.
            runtime_config: Resolved device, distribution, and logging configuration.
            logger: Existing logger or W&B run reused by training.
        """
        self.training_config = training_config
        self.diffusion_config = diffusion_config
        self.device = device
        self.random_state = random_state
        self.runtime_config = runtime_config or TabFORGERuntimeConfig(device=str(device))
        self.logger = logger

    def fit(
        self,
        decoder: TabFORGEDecoder,
        denoiser: VariableColumnDenoiser,
        clean_latents: np.ndarray,
        targets: np.ndarray,
        *,
        has_target: bool,
        target_kind: str | None,
        conditional_masking: bool,
        target_cardinality: int,
        numerical_count: int,
        categorical_cardinalities: tuple[int, ...],
        latent_mean: np.ndarray | torch.Tensor,
        latent_std: np.ndarray | torch.Tensor,
        validation_latents: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        target_prediction: bool = False,
    ) -> TabFORGETrainingState:
        """Train decoder and diffusion components with optional validation.

        Args:
            decoder: Schema-aware latent decoder.
            denoiser: Noise-conditioned latent denoiser.
            clean_latents: Normalized clean latent grid for training.
            targets: Dense reconstruction targets aligned with latent rows.
            has_target: Whether the fitted latent layout includes a target token.
            target_kind: Classification, regression, or ``None``.
            conditional_masking: Whether observation masks are sampled during training.
            target_cardinality: Number of fitted classification target values.
            numerical_count: Number of leading numerical reconstruction values.
            categorical_cardinalities: Ordered categorical reconstruction widths.
            latent_mean: Fitted latent mean broadcast across rows.
            latent_std: Fitted latent standard deviation broadcast across rows.
            validation_latents: Optional normalized validation latent grid.
            validation_targets: Optional dense validation reconstruction targets.
            target_prediction: Whether the lightweight supervised head is trained.
        """
        fit_args = self._fit_arguments(
            has_target,
            target_kind,
            conditional_masking,
            target_cardinality,
            numerical_count,
            categorical_cardinalities,
            latent_mean,
            latent_std,
            validation_latents,
            validation_targets,
            target_prediction,
            self.logger,
        )
        return train_components(decoder, denoiser, clean_latents, targets, **fit_args)

    def _fit_arguments(
        self,
        has_target: bool,
        target_kind: str | None,
        conditional_masking: bool,
        target_cardinality: int,
        numerical_count: int,
        categorical_cardinalities: tuple[int, ...],
        latent_mean: np.ndarray | torch.Tensor,
        latent_std: np.ndarray | torch.Tensor,
        validation_latents: np.ndarray | None,
        validation_targets: np.ndarray | None,
        target_prediction: bool,
        logger: Any,
    ) -> dict[str, Any]:
        """Build arguments owned by the configured trainer instance.

        Args:
            has_target: Whether the fitted latent layout includes a target token.
            target_kind: Classification, regression, or ``None``.
            conditional_masking: Whether observation masks are sampled during training.
            target_cardinality: Number of fitted classification target values.
            numerical_count: Number of leading numerical reconstruction values.
            categorical_cardinalities: Ordered categorical reconstruction widths.
            latent_mean: Fitted latent mean broadcast across rows.
            latent_std: Fitted latent standard deviation broadcast across rows.
            validation_latents: Optional normalized validation latent grid.
            validation_targets: Optional dense validation reconstruction targets.
            target_prediction: Whether the lightweight supervised head is trained.
            logger: Existing logger or W&B run reused by training.
        """
        return {
            "training_config": self.training_config,
            "diffusion_config": self.diffusion_config,
            "device": self.device,
            "random_state": self.random_state,
            "has_target": has_target,
            "target_kind": target_kind,
            "conditional_masking": conditional_masking,
            "target_cardinality": target_cardinality,
            "numerical_count": numerical_count,
            "categorical_cardinalities": categorical_cardinalities,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "validation_latents": validation_latents,
            "validation_targets": validation_targets,
            "target_prediction": target_prediction,
            "runtime_config": self.runtime_config,
            "logger": logger,
        }


# ================================================================
# =                                                              =
# =                     Model fitting                            =
# =                                                              =
# ================================================================


@dataclass
class _LossOptions:
    """Group schema and diffusion settings used by phase losses."""

    diffusion_config: TabFORGEDiffusionConfig
    has_target: bool
    target_kind: str | None
    target_cardinality: int
    numerical_count: int
    categorical_cardinalities: tuple[int, ...]
    target_loss_weight: float
    target_prediction: bool
    latent_mean: torch.Tensor
    latent_std: torch.Tensor


@dataclass
class _TrainingOptions:
    """Group immutable options shared throughout one training run."""

    training_config: TabFORGETrainingConfig
    runtime_config: TabFORGERuntimeConfig
    device: torch.device
    random_state: int | None
    loss: _LossOptions
    conditional_masking: bool
    logger: TrainingLogger


@dataclass
class _TrainingData:
    """Hold CPU tensors for training and optional validation."""

    clean: torch.Tensor
    target: torch.Tensor
    validation_clean: torch.Tensor | None
    validation_target: torch.Tensor | None
    train_rows: int

    @property
    def has_validation(self) -> bool:
        """Return whether an explicit validation split is available."""
        return self.validation_clean is not None


@dataclass
class _TrainingModels:
    """Group the decoder, denoiser, and preconditioned diffusion model."""

    decoder: TabFORGEDecoder
    denoiser: VariableColumnDenoiser
    diffusion: EDMPreconditioner


@dataclass
class _PhasePair:
    """Associate decoder and diffusion resources with their phases."""

    decoder: Any
    diffusion: Any


@dataclass
class _StepScheduler:
    """Reduce one optimizer learning rate using optimizer-step patience."""

    optimizer: torch.optim.Optimizer
    config: TabFORGESchedulerConfig
    best: float | None = None
    last_improvement_step: int = 0
    last_reduce_step: int = 0

    def step(self, metric: float, optimizer_step: int) -> None:
        """Update the learning rate after a validation measurement.

        Args:
            metric: Latest validation metric supplied to the scheduler.
            optimizer_step: Completed global optimizer-step index.
        """
        if self.config.patience_steps is None:
            return
        if self.best is None or metric < self.best - self.config.min_delta:
            self.best = metric
            self.last_improvement_step = optimizer_step
            return
        reference_step = max(self.last_improvement_step, self.last_reduce_step)
        if optimizer_step - reference_step < self.config.patience_steps:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = max(self.config.min_lr, group["lr"] * self.config.factor)
        self.last_reduce_step = optimizer_step


@dataclass
class _RandomSources:
    """Hold deterministic torch and NumPy sampling sources."""

    torch_generator: torch.Generator
    numpy_generator: np.random.Generator
    masking_probabilities: dict[str, float]
    row_order: torch.Tensor | None = None
    row_offset: int = 0


@dataclass
class _BestComponents:
    """Track component weights selected by validation or training loss."""

    decoder: dict[str, Any]
    denoiser: dict[str, Any]
    decoder_loss: float
    diffusion_loss: float


@dataclass
class _TrainingContext:
    """Collect mutable and immutable state for the phase loop."""

    options: _TrainingOptions
    data: _TrainingData
    models: _TrainingModels
    optimizers: _PhasePair
    schedulers: _PhasePair
    random: _RandomSources
    state: TabFORGETrainingState
    best: _BestComponents
    progress: _TrainingProgress
    world_size: int
    validation_interval: int


@dataclass
class _LossBatch:
    """Hold one sampled batch and its diffusion perturbation."""

    clean: torch.Tensor
    target: torch.Tensor
    sigma: torch.Tensor
    noisy: torch.Tensor
    observed_mask: torch.Tensor


@dataclass
class _TrainingProgress:
    """Render a compact rank-zero training progress bar."""

    total: int
    enabled: bool

    def update(self, state: TabFORGETrainingState, loss: float) -> None:
        """Display the completed optimizer steps and active phase.

        Args:
            state: Mutable training state or serialized component state.
            loss: Scalar candidate loss value.
        """
        if not self.enabled:
            return
        completed = min(state.global_step, self.total)
        refresh_interval = max(1, self.total // _PROGRESS_REFRESH_COUNT)
        if completed != self.total and completed % refresh_interval:
            return
        filled = _PROGRESS_BAR_WIDTH * completed // self.total
        bar = f"{'=' * filled}{'.' * (_PROGRESS_BAR_WIDTH - filled)}"
        ending = "\n" if completed == self.total else "\r"
        print(
            f"TabFORGE training [{bar}] {completed}/{self.total} " f"phase={state.phase} loss={loss:.4f}",
            end=ending,
            file=sys.stderr,
            flush=True,
        )


def train_components(
    decoder: TabFORGEDecoder,
    denoiser: VariableColumnDenoiser,
    clean_latents: np.ndarray,
    targets: np.ndarray,
    *,
    training_config: TabFORGETrainingConfig,
    diffusion_config: TabFORGEDiffusionConfig,
    device: torch.device,
    random_state: int | None,
    has_target: bool,
    target_kind: str | None,
    conditional_masking: bool = True,
    target_cardinality: int,
    numerical_count: int,
    categorical_cardinalities: tuple[int, ...],
    latent_mean: np.ndarray | torch.Tensor,
    latent_std: np.ndarray | torch.Tensor,
    validation_latents: np.ndarray | None = None,
    validation_targets: np.ndarray | None = None,
    target_prediction: bool = False,
    runtime_config: TabFORGERuntimeConfig | None = None,
    logger: Any = None,
) -> TabFORGETrainingState:
    """Run phase-based component training and return progress metadata.

    Args:
        decoder: Schema-aware latent decoder.
        denoiser: Noise-conditioned latent denoiser.
        clean_latents: Normalized clean latent grid for training.
        targets: Dense reconstruction targets aligned with latent rows.
        training_config: Resolved optimizer and phase configuration.
        diffusion_config: Resolved EDM training and sampling configuration.
        device: Torch device used by tensors and neural components.
        random_state: Optional seed for deterministic random streams.
        has_target: Whether the fitted latent layout includes a target token.
        target_kind: Classification, regression, or ``None``.
        conditional_masking: Whether observation masks are sampled during training.
        target_cardinality: Number of fitted classification target values.
        numerical_count: Number of leading numerical reconstruction values.
        categorical_cardinalities: Ordered categorical reconstruction widths.
        latent_mean: Fitted latent mean broadcast across rows.
        latent_std: Fitted latent standard deviation broadcast across rows.
        validation_latents: Optional normalized validation latent grid.
        validation_targets: Optional dense validation reconstruction targets.
        target_prediction: Whether the lightweight supervised head is trained.
        runtime_config: Resolved device, distribution, and logging configuration.
        logger: Existing logger or W&B run reused by training.
    """
    runtime_config = runtime_config or TabFORGERuntimeConfig(device=str(device))
    trace_logger = TrainingLogger(runtime_config, logger)
    options = _build_training_options(
        training_config,
        diffusion_config,
        runtime_config,
        device,
        random_state,
        has_target,
        target_kind,
        conditional_masking,
        target_cardinality,
        numerical_count,
        categorical_cardinalities,
        latent_mean,
        latent_std,
        target_prediction,
        trace_logger,
    )
    try:
        context = _prepare_training_context(
            decoder,
            denoiser,
            clean_latents,
            targets,
            validation_latents,
            validation_targets,
            options,
        )
        _run_training_phases(context)
        return _finalize_training(context)
    finally:
        trace_logger.finish()


def _build_training_options(
    training: TabFORGETrainingConfig,
    diffusion: TabFORGEDiffusionConfig,
    runtime: TabFORGERuntimeConfig,
    device: torch.device,
    random_state: int | None,
    has_target: bool,
    target_kind: str | None,
    conditional_masking: bool,
    target_cardinality: int,
    numerical_count: int,
    cardinalities: tuple[int, ...],
    latent_mean: np.ndarray | torch.Tensor,
    latent_std: np.ndarray | torch.Tensor,
    target_prediction: bool,
    logger: TrainingLogger,
) -> _TrainingOptions:
    """Group public training arguments for focused phase helpers.

    Args:
        training: Serialized training metadata or resolved training configuration.
        diffusion: Resolved diffusion configuration.
        runtime: Resolved runtime configuration.
        device: Torch device used by tensors and neural components.
        random_state: Optional seed for deterministic random streams.
        has_target: Whether the fitted latent layout includes a target token.
        target_kind: Classification, regression, or ``None``.
        conditional_masking: Whether observation masks are sampled during training.
        target_cardinality: Number of fitted classification target values.
        numerical_count: Number of leading numerical reconstruction values.
        cardinalities: Ordered categorical reconstruction widths.
        latent_mean: Fitted latent mean broadcast across rows.
        latent_std: Fitted latent standard deviation broadcast across rows.
        target_prediction: Whether the lightweight supervised head is trained.
        logger: Existing logger or W&B run reused by training.
    """
    # Decoder inputs return to the original TabPFN embedding scale.
    mean = torch.as_tensor(latent_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(latent_std, dtype=torch.float32, device=device)
    loss = _LossOptions(
        diffusion,
        has_target,
        target_kind,
        target_cardinality,
        numerical_count,
        cardinalities,
        training.target_loss_weight,
        target_prediction,
        mean,
        std,
    )
    return _TrainingOptions(training, runtime, device, random_state, loss, conditional_masking, logger)


def _prepare_training_context(
    decoder: TabFORGEDecoder,
    denoiser: VariableColumnDenoiser,
    clean_latents: np.ndarray,
    targets: np.ndarray,
    validation_latents: np.ndarray | None,
    validation_targets: np.ndarray | None,
    options: _TrainingOptions,
) -> _TrainingContext:
    """Build the state shared by phase-focused training helpers.

    Args:
        decoder: Schema-aware latent decoder.
        denoiser: Noise-conditioned latent denoiser.
        clean_latents: Normalized clean latent grid for training.
        targets: Dense reconstruction targets aligned with latent rows.
        validation_latents: Optional normalized validation latent grid.
        validation_targets: Optional dense validation reconstruction targets.
        options: Resolved immutable options for the training run.
    """
    _validate_runtime_config(options.runtime_config)
    distributed, rank, world_size = _distributed_layout()
    data = _prepare_training_data(
        clean_latents,
        targets,
        validation_latents,
        validation_targets,
        options.device,
        distributed,
        rank,
        world_size,
    )
    models = _prepare_training_models(decoder, denoiser, options, distributed)
    optimizers = _prepare_phase_optimizers(models, options.training_config)
    schedulers = _prepare_phase_schedulers(optimizers, options.training_config, data.has_validation)
    random_sources = _prepare_random_sources(options, rank)
    state, best = _prepare_training_state(models)
    progress = _TrainingProgress(
        total=options.training_config.max_steps,
        enabled=rank == 0,
    )
    interval = _validation_interval(
        data.train_rows,
        options.training_config,
        options.runtime_config.gradient_accumulation,
    )
    return _TrainingContext(
        options,
        data,
        models,
        optimizers,
        schedulers,
        random_sources,
        state,
        best,
        progress,
        world_size,
        interval,
    )


def _distributed_layout() -> tuple[bool, int, int]:
    """Resolve active distributed status, rank, and world size."""
    distributed = _distributed_active()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    return distributed, rank, world_size


def _prepare_training_data(
    clean_latents: np.ndarray,
    targets: np.ndarray,
    validation_latents: np.ndarray | None,
    validation_targets: np.ndarray | None,
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
) -> _TrainingData:
    """Shard CPU embedding arrays before moving each rank's data to its device.

    Args:
        clean_latents: Normalized clean latent grid for training.
        targets: Dense reconstruction targets aligned with latent rows.
        validation_latents: Optional normalized validation latent grid.
        validation_targets: Optional dense validation reconstruction targets.
        device: Torch device used by tensors and neural components.
        distributed: Whether an active multi-rank process group is used.
        rank: Current distributed rank.
        world_size: Number of participating distributed ranks.
    """
    if (validation_latents is None) != (validation_targets is None):
        raise ValueError("validation_latents and validation_targets must be provided together")
    clean = torch.as_tensor(clean_latents, dtype=torch.float32)
    target = torch.as_tensor(targets, dtype=torch.float32)
    validation_clean = _optional_tensor(validation_latents)
    validation_target = _optional_tensor(validation_targets)
    if validation_clean is not None:
        _validate_validation_tensors(clean, validation_clean, validation_target)
    train_rows = len(clean)
    if distributed:
        indices = torch.arange(rank, train_rows, world_size)
        clean = clean.index_select(0, indices)
        target = target.index_select(0, indices)
        if len(clean) == 0:
            raise ValueError("Distributed training requires at least one local row per rank")
    return _TrainingData(
        clean.to(device),
        target.to(device),
        None if validation_clean is None else validation_clean.to(device),
        None if validation_target is None else validation_target.to(device),
        train_rows,
    )


def _optional_tensor(values: np.ndarray | None) -> torch.Tensor | None:
    """Convert an optional validation array to a CPU tensor.

    Args:
        values: Values handled by the current transformation.
    """
    return None if values is None else torch.as_tensor(values, dtype=torch.float32)


def _validate_validation_tensors(clean: torch.Tensor, validation: torch.Tensor, target: torch.Tensor) -> None:
    """Validate validation row counts and latent shape compatibility.

    Args:
        clean: Clean latent batch.
        validation: Validation latent grid.
        target: Target values or tensor aligned with the current rows.
    """
    if len(validation) < 1:
        raise ValueError("validation data must contain at least one row")
    if len(validation) != len(target):
        raise ValueError("validation latents and targets must contain the same number of rows")
    if tuple(validation.shape[1:]) != tuple(clean.shape[1:]):
        raise ValueError("training and validation latents must have matching token shapes")


def _prepare_training_models(
    decoder: TabFORGEDecoder,
    denoiser: VariableColumnDenoiser,
    options: _TrainingOptions,
    distributed: bool,
) -> _TrainingModels:
    """Move components to the device and synchronize distributed weights.

    Args:
        decoder: Schema-aware latent decoder.
        denoiser: Noise-conditioned latent denoiser.
        options: Resolved immutable options for the training run.
        distributed: Whether an active multi-rank process group is used.
    """
    decoder.to(options.device).train()
    denoiser.to(options.device).train()
    if distributed:
        _broadcast_module_state(decoder)
        _broadcast_module_state(denoiser)
    diffusion = EDMPreconditioner(denoiser, options.loss.diffusion_config.sigma_data).to(options.device)
    return _TrainingModels(decoder, denoiser, diffusion)


def _prepare_phase_optimizers(models: _TrainingModels, config: TabFORGETrainingConfig) -> _PhasePair:
    """Build independent optimizers for both training phases.

    Args:
        models: Decoder, denoiser, and EDM model bundle.
        config: Validated configuration or serialized configuration mapping.
    """
    decoder = _make_optimizer(
        config.decoder_optimizer,
        models.decoder.parameters(),
        config.decoder_lr,
        config.decoder_weight_decay,
    )
    diffusion = _make_optimizer(
        config.diffusion_optimizer,
        models.denoiser.parameters(),
        config.diffusion_lr,
        config.diffusion_weight_decay,
    )
    return _PhasePair(decoder, diffusion)


def _prepare_phase_schedulers(optimizers: _PhasePair, config: TabFORGETrainingConfig, enabled: bool) -> _PhasePair:
    """Build validation schedulers for both phase optimizers when enabled.

    Args:
        optimizers: Decoder and diffusion optimizer bundle.
        config: Validated configuration or serialized configuration mapping.
        enabled: Whether the optional operation is active.
    """
    decoder = _make_scheduler(optimizers.decoder, config.decoder_scheduler, enabled=enabled)
    diffusion = _make_scheduler(optimizers.diffusion, config.diffusion_scheduler, enabled=enabled)
    return _PhasePair(decoder, diffusion)


def _prepare_random_sources(options: _TrainingOptions, rank: int) -> _RandomSources:
    """Build rank-specific random sources without changing sample order.

    Args:
        options: Resolved immutable options for the training run.
        rank: Current distributed rank.
    """
    seed = 0 if options.random_state is None else int(options.random_state)
    torch_rng = torch.Generator(device=options.device)
    torch_rng.manual_seed(seed + rank)
    numpy_seed = None if options.random_state is None else seed + rank
    numpy_rng = np.random.default_rng(numpy_seed)
    probabilities = {"unmasked": 1.0, "target": 0.0, "random": 0.0}
    if options.loss.target_prediction:
        probabilities = {"unmasked": 0.0, "target": 1.0, "random": 0.0}
    elif options.conditional_masking:
        probabilities = options.training_config.normalized_masking_probabilities(has_target=options.loss.has_target)
    return _RandomSources(torch_rng, numpy_rng, probabilities)


def _prepare_training_state(
    models: _TrainingModels,
) -> tuple[TabFORGETrainingState, _BestComponents]:
    """Initialize progress metadata and best component snapshots.

    Args:
        models: Decoder, denoiser, and EDM model bundle.
    """
    state = TabFORGETrainingState()
    best = _BestComponents(
        copy.deepcopy(models.decoder.state_dict()),
        copy.deepcopy(models.denoiser.state_dict()),
        float("inf") if state.best_decoder_loss is None else state.best_decoder_loss,
        float("inf") if state.best_diffusion_loss is None else state.best_diffusion_loss,
    )
    return state, best


def _validation_interval(train_rows: int, config: TabFORGETrainingConfig, accumulation: int) -> int:
    """Convert epoch-based validation cadence to training steps.

    Args:
        train_rows: Number of rank-local training rows.
        config: Validated configuration or serialized configuration mapping.
        accumulation: Gradient-accumulation batches per optimizer update.
    """
    # Epoch length follows complete training batches.
    steps_per_epoch = max(1, train_rows // config.batch_size // accumulation)
    return steps_per_epoch * config.validation_every_n_epochs


def _run_training_phases(context: _TrainingContext) -> None:
    """Run each enabled phase within its configured step budget.

    Args:
        context: Mutable state for the current training run.
    """
    phase_steps = _phase_steps(context.options.training_config)
    phases = (("diffusion", phase_steps[0]), ("decoder", phase_steps[1]))
    for phase_index, (phase, count) in enumerate(phases):
        _run_training_phase(context, phase, count)
        # The following phase consumes the best completed component.
        _restore_best_component(context, phase)
        context.state.phase_step = 0
        context.state.epoch = phase_index + 1
        if phase == "diffusion" and phase_steps[1] == 0:
            context.state.phase = "complete"


def _run_training_phase(context: _TrainingContext, phase: str, count: int) -> None:
    """Run one component training phase.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        count: Optimizer-step budget for the active phase.
    """
    context.state.phase = phase
    _set_trainable(context.models.decoder, phase == "decoder")
    _set_trainable(context.models.denoiser, phase == "diffusion")
    optimizer = _phase_value(context.optimizers, phase)
    scheduler = _phase_value(context.schedulers, phase)
    final_loss = None
    for _ in range(count):
        final_loss = _run_training_step(context, phase, count, optimizer, scheduler)
    if final_loss is not None and not context.data.has_validation:
        _update_best_component(context, phase, final_loss, force=True)


def _run_training_step(
    context: _TrainingContext,
    phase: str,
    count: int,
    optimizer: torch.optim.Optimizer,
    scheduler: _StepScheduler | None,
) -> float:
    """Run one optimizer update and its state bookkeeping.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        count: Optimizer-step budget for the active phase.
        optimizer: Optimizer for the active training phase.
        scheduler: Optional optimizer-step plateau scheduler.
    """
    losses = _backward_phase_loss(context, phase, optimizer)
    _step_phase_optimizer(context, phase, optimizer)
    scalar = _distributed_mean_loss(context, losses)
    phase_step = _record_training_step(context, phase)
    validation_loss = _step_validation_scheduler(context, phase, phase_step, count, scheduler)
    if validation_loss is not None:
        _update_best_component(context, phase, validation_loss)
    _log_training_step(context, phase, phase_step, scalar, validation_loss)
    context.progress.update(context.state, scalar)
    return scalar


def _backward_phase_loss(
    context: _TrainingContext,
    phase: str,
    optimizer: torch.optim.Optimizer,
) -> list[float]:
    """Accumulate gradients from the configured number of sampled batches.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        optimizer: Optimizer for the active training phase.
    """
    optimizer.zero_grad(set_to_none=True)
    losses = []
    accumulation = context.options.runtime_config.gradient_accumulation
    for _ in range(accumulation):
        batch = _sample_training_batch(context)
        loss = _phase_loss(context, phase, batch)
        losses.append(float(loss.detach().cpu()))
        (loss / accumulation).backward()
    return losses


def _sample_training_batch(context: _TrainingContext) -> _LossBatch:
    """Sample rows, noise levels, noise, and observation masks in stable order.

    Args:
        context: Mutable state for the current training run.
    """
    data = context.data
    indices = _next_training_indices(context)
    clean = data.clean.index_select(0, indices)
    target = data.target.index_select(0, indices)
    return _build_loss_batch(
        context,
        clean,
        target,
        context.random.torch_generator,
        context.random.numpy_generator,
    )


def _build_loss_batch(
    context: _TrainingContext,
    clean: torch.Tensor,
    target: torch.Tensor,
    torch_generator: torch.Generator,
    numpy_generator: np.random.Generator,
) -> _LossBatch:
    """Build a noisy batch with clean observed tokens for conditional tasks.

    Args:
        context: Mutable state for the current training run.
        clean: Clean latent batch.
        target: Target values or tensor aligned with the current rows.
        torch_generator: Torch random stream for tensor sampling.
        numpy_generator: NumPy random stream for masking-mode selection.
    """
    sigma = _sample_sigma(context, len(clean), torch_generator)
    noise = torch.randn(clean.shape, generator=torch_generator, device=context.options.device)
    noisy = clean + sigma.reshape(-1, 1, 1) * noise
    observed_mask = _sample_batch_mask(context, len(clean), numpy_generator)
    if context.options.conditional_masking:
        noisy = torch.where(observed_mask.unsqueeze(-1), clean, noisy)
    return _LossBatch(clean, target, sigma, noisy, observed_mask)


def _next_training_indices(context: _TrainingContext) -> torch.Tensor:
    """Return the next shuffled batch without replacement within an epoch.

    Args:
        context: Mutable state for the current training run.
    """
    rows = len(context.data.clean)
    random_sources = context.random
    batch_size = min(context.options.training_config.batch_size, rows)
    if random_sources.row_order is None or random_sources.row_offset + batch_size > rows:
        random_sources.row_order = torch.randperm(
            rows,
            generator=random_sources.torch_generator,
            device=context.options.device,
        )
        random_sources.row_offset = 0
    start = random_sources.row_offset
    stop = start + batch_size
    random_sources.row_offset = stop
    return random_sources.row_order[start:stop]


def _sample_sigma(context: _TrainingContext, rows: int, rng: torch.Generator) -> torch.Tensor:
    """Sample clamped log-normal EDM noise levels.

    Args:
        context: Mutable state for the current training run.
        rows: Number of rows to sample or process.
        rng: Torch random stream used for tensor sampling.
    """
    config = context.options.loss.diffusion_config
    values = torch.randn(rows, generator=rng, device=context.options.device) * config.p_std + config.p_mean
    sigma_max = config.sigma_max
    if context.options.loss.target_prediction:
        sigma_max = max(config.sigma_min, min(config.sigma_init, config.sigma_max))
    return torch.exp(values).clamp(config.sigma_min, sigma_max)


def _sample_batch_mask(
    context: _TrainingContext,
    rows: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample observation masks for one training or validation batch.

    Args:
        context: Mutable state for the current training run.
        rows: Number of rows to sample or process.
        rng: Torch random stream used for tensor sampling.
    """
    return _sample_observation_mask(
        rows,
        context.data.clean.shape[1],
        has_target=context.options.loss.has_target,
        probabilities=context.random.masking_probabilities,
        rng=rng,
        device=context.options.device,
    )


def _phase_loss(context: _TrainingContext, phase: str, batch: _LossBatch) -> torch.Tensor:
    """Compute the active decoder or diffusion objective.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        batch: Sampled clean/noisy tensors for one loss computation.
    """
    if phase == "decoder" and context.options.loss.target_prediction:
        return _prediction_target_loss(context, batch.clean, batch.target)
    # Both phases use the same noisy denoising path.
    denoised = context.models.diffusion(batch.noisy, batch.sigma, batch.observed_mask)
    if context.options.conditional_masking:
        denoised = torch.where(batch.observed_mask.unsqueeze(-1), batch.clean, denoised)
    if phase == "decoder":
        return _reconstruction_loss(context, denoised, batch.target)
    config = context.options.loss.diffusion_config
    # EDM weighting keeps unsupervision and supervised loss scales aligned.
    loss_mask = ~batch.observed_mask if context.options.conditional_masking else None
    loss = edm_weighted_mse(denoised, batch.clean, batch.sigma, config.sigma_data, loss_mask=loss_mask)
    return loss


def _reconstruction_loss(
    context: _TrainingContext,
    latent: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute mixed-schema reconstruction loss from latent tokens.

    Args:
        context: Mutable state for the current training run.
        latent: Latent grid used by the current loss or decode step.
        target: Target values or tensor aligned with the current rows.
    """
    options = context.options.loss
    # The decoder was designed for denormalized TabPFN embeddings.
    recovered = latent * options.latent_std + options.latent_mean
    decoded = context.models.decoder(recovered)
    losses = mixed_feature_loss(
        decoded,
        target,
        numerical_count=options.numerical_count,
        categorical_cardinalities=options.categorical_cardinalities,
        target_kind=options.target_kind,
    )
    return losses["reconstruction"] + options.target_loss_weight * losses["target"]


def _prediction_target_loss(
    context: _TrainingContext,
    latent: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Train the lightweight predictor head on clean normalized target tokens.

    Args:
        context: Mutable state for the current training run.
        latent: Latent grid used by the current loss or decode step.
        target: Target values or tensor aligned with the current rows.
    """
    options = context.options.loss
    prediction = context.models.decoder.predict_target(latent)
    width = options.target_cardinality if options.target_kind == "classification" else 1
    target_values = target[:, -width:]
    if options.target_kind == "classification":
        return F.cross_entropy(prediction, target_values.argmax(dim=1))
    return F.mse_loss(prediction.reshape(-1, 1), target_values)


def _step_phase_optimizer(
    context: _TrainingContext,
    phase: str,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Average gradients, clip them, and update the active component.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        optimizer: Optimizer for the active training phase.
    """
    module = context.models.denoiser if phase == "diffusion" else context.models.decoder
    _average_gradients(module, context.world_size)
    if context.options.training_config.gradient_clip_value:
        # Clip the active component before its optimizer update.
        nn.utils.clip_grad_norm_(module.parameters(), context.options.training_config.gradient_clip_value)
    optimizer.step()


def _distributed_mean_loss(context: _TrainingContext, losses: list[float]) -> float:
    """Average accumulated loss values across active ranks.

    Args:
        context: Mutable state for the current training run.
        losses: Per-accumulation scalar loss values.
    """
    scalar = torch.tensor(float(np.mean(losses)), device=context.options.device)
    if context.world_size > 1:
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        scalar = scalar / context.world_size
    return float(scalar.cpu())


def _step_validation_scheduler(
    context: _TrainingContext,
    phase: str,
    step: int,
    count: int,
    scheduler: _StepScheduler | None,
) -> float | None:
    """Evaluate validation and advance the active plateau scheduler.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        step: Completed global optimizer-step index.
        count: Optimizer-step budget for the active phase.
        scheduler: Optional optimizer-step plateau scheduler.
    """
    if not context.data.has_validation:
        return None
    if step % context.validation_interval and step != count:
        return None
    validation_loss = _evaluate_validation_loss(context, phase)
    if scheduler is not None:
        scheduler.step(validation_loss, step)
    return validation_loss


def _update_best_component(
    context: _TrainingContext,
    phase: str,
    loss: float,
    *,
    force: bool = False,
) -> None:
    """Snapshot an active component when its selection loss improves.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        loss: Scalar candidate loss value.
        force: Record the candidate even without a measured improvement.
    """
    if phase == "diffusion" and (force or loss < context.best.diffusion_loss):
        context.best.diffusion_loss = loss
        context.best.denoiser = copy.deepcopy(context.models.denoiser.state_dict())
    if phase == "decoder" and (force or loss < context.best.decoder_loss):
        context.best.decoder_loss = loss
        context.best.decoder = copy.deepcopy(context.models.decoder.state_dict())


def _restore_best_component(context: _TrainingContext, phase: str) -> None:
    """Restore the best phase checkpoint before the next phase starts.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
    """
    if phase == "diffusion":
        context.models.denoiser.load_state_dict(context.best.denoiser)
    else:
        context.models.decoder.load_state_dict(context.best.decoder)


def _record_training_step(context: _TrainingContext, phase: str) -> int:
    """Advance training progress counters.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
    """
    context.state.global_step += 1
    if phase == "decoder":
        context.state.decoder_optimizer_step += 1
        context.state.phase_step = context.state.decoder_optimizer_step
    else:
        context.state.diffusion_optimizer_step += 1
        context.state.phase_step = context.state.diffusion_optimizer_step
    return context.state.phase_step


def _log_training_step(
    context: _TrainingContext,
    phase: str,
    phase_step: int,
    train_loss: float,
    validation_loss: float | None,
) -> None:
    """Write the compact optimizer trace when enabled.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        phase_step: Completed optimizer step within the active phase.
        train_loss: Latest averaged training loss.
        validation_loss: Optional complete-split validation loss.
    """
    if not context.options.logger.enabled:
        return
    values = {
        "training/optimizer_step": context.state.global_step,
        "training/decoder_optimizer_step": context.state.decoder_optimizer_step,
        "training/diffusion_optimizer_step": context.state.diffusion_optimizer_step,
        "training/phase": phase,
        "training/phase_optimizer_step": phase_step,
        "training/decoder_learning_rate": _current_learning_rate(context.optimizers.decoder),
        "training/diffusion_learning_rate": _current_learning_rate(context.optimizers.diffusion),
    }
    values[f"training/{'reconstruction' if phase == 'decoder' else 'diffusion'}_loss"] = train_loss
    if validation_loss is not None:
        values[f"valid/{'reconstruction' if phase == 'decoder' else 'diffusion'}_loss"] = validation_loss
    context.options.logger.log(values, step=context.state.global_step)


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the first parameter-group learning rate.

    Args:
        optimizer: Optimizer for the active training phase.
    """
    return float(optimizer.param_groups[0]["lr"])


def _finalize_training(context: _TrainingContext) -> TabFORGETrainingState:
    """Restore best components and finalize progress metadata.

    Args:
        context: Mutable state for the current training run.
    """
    context.state.phase = "complete"
    context.models.denoiser.load_state_dict(context.best.denoiser)
    context.models.decoder.load_state_dict(context.best.decoder)
    diffusion_steps, decoder_steps = _phase_steps(context.options.training_config)
    context.state.best_diffusion_loss = context.best.diffusion_loss if diffusion_steps else None
    context.state.best_decoder_loss = context.best.decoder_loss if decoder_steps else None
    return context.state


def _phase_value(values: _PhasePair, phase: str) -> Any:
    """Select the resource associated with an active phase.

    Args:
        values: Values handled by the current transformation.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
    """
    return values.diffusion if phase == "diffusion" else values.decoder


# ================================================================
# =                                                              =
# =                    Training utilities                        =
# =                                                              =
# ================================================================


def _distributed_active() -> bool:
    """Return whether multi-rank torch distribution is active."""
    return distributed_active()


def _validate_runtime_config(config: TabFORGERuntimeConfig) -> None:
    """Validate the direct trainer's execution boundary before training.

    Args:
        config: Validated configuration or serialized configuration mapping.
    """
    if config.strategy == "ddp" and not _distributed_active():
        raise RuntimeError("A distributed process group is required for the requested training strategy")


def _set_trainable(module: nn.Module, enabled: bool) -> None:
    """Enable gradients only for the component trained in the active phase.

    Args:
        module: Neural module affected by the operation.
        enabled: Whether the optional operation is active.
    """
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _broadcast_module_state(module: nn.Module) -> None:
    """Broadcast initial component state from the primary rank.

    Args:
        module: Neural module affected by the operation.
    """
    for tensor in module.state_dict().values():
        dist.broadcast(tensor, src=0)


def _average_gradients(module: nn.Module, world_size: int) -> None:
    """Average active component gradients across ranks.

    Args:
        module: Neural module affected by the operation.
        world_size: Number of participating distributed ranks.
    """
    if world_size == 1:
        return
    for parameter in module.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def _phase_steps(config: TabFORGETrainingConfig) -> tuple[int, int]:
    """Allocate the total step budget between diffusion and decoder phases.

    Args:
        config: Validated configuration or serialized configuration mapping.
    """
    total = config.max_steps
    ratio_total = config.decoder_phase_ratio + config.diffusion_phase_ratio
    diffusion_steps = int(round(total * config.diffusion_phase_ratio / ratio_total))
    diffusion_steps = min(total, max(0, diffusion_steps))
    return diffusion_steps, total - diffusion_steps


def _make_optimizer(
    name: str,
    parameters: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build a configured optimizer for one trainable component.

    Args:
        name: Registered estimator or optimizer name.
        parameters: Trainable parameters owned by one optimizer.
        lr: Initial optimizer learning rate.
        weight_decay: Optimizer weight-decay coefficient.
    """
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        # The diffusion optimizer uses a slightly faster second-moment response.
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    raise ValueError(f"Unsupported optimizer {name!r}")


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TabFORGESchedulerConfig,
    *,
    enabled: bool,
) -> _StepScheduler | None:
    """Create a plateau scheduler only for explicit validation runs.

    Args:
        optimizer: Optimizer for the active training phase.
        config: Validated configuration or serialized configuration mapping.
        enabled: Whether the optional operation is active.
    """
    if not enabled or config.patience_steps is None:
        return None
    return _StepScheduler(optimizer, config)


# ================================================================
# =                                                              =
# =                     Model validation                         =
# =                                                              =
# ================================================================


def _evaluate_validation_loss(context: _TrainingContext, phase: str) -> float:
    """Evaluate the active phase over the complete validation split.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
    """
    # === Switch components to evaluation mode ===
    decoder_was_training = context.models.decoder.training
    denoiser_was_training = context.models.denoiser.training
    context.models.decoder.eval()
    context.models.denoiser.eval()
    try:
        # Fresh draws preserve the stochastic validation objective.
        value = _validation_split_loss(
            context,
            phase,
            context.random.torch_generator,
            context.random.numpy_generator,
        )
        if context.world_size > 1:
            scalar = torch.tensor(value, device=context.options.device)
            dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
            value = float((scalar / context.world_size).cpu())
        return value
    finally:
        # === Restore the active training modes ===
        context.models.decoder.train(decoder_was_training)
        context.models.denoiser.train(denoiser_was_training)


def _validation_split_loss(
    context: _TrainingContext,
    phase: str,
    torch_generator: torch.Generator,
    numpy_generator: np.random.Generator,
) -> float:
    """Accumulate a row-weighted loss over the validation split.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        torch_generator: Torch random stream for tensor sampling.
        numpy_generator: NumPy random stream for masking-mode selection.
    """
    clean = context.data.validation_clean
    target = context.data.validation_target
    batch_size = context.options.training_config.batch_size
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(clean), batch_size):
            clean_batch = clean[start : start + batch_size]
            target_batch = target[start : start + batch_size]
            loss = _validation_batch_loss(
                context,
                phase,
                clean_batch,
                target_batch,
                torch_generator,
                numpy_generator,
            )
            total_loss += float(loss.cpu()) * len(clean_batch)
    return total_loss / len(clean)


def _validation_batch_loss(
    context: _TrainingContext,
    phase: str,
    clean: torch.Tensor,
    target: torch.Tensor,
    torch_generator: torch.Generator,
    numpy_generator: np.random.Generator,
) -> torch.Tensor:
    """Compute one stochastic validation batch loss.

    Args:
        context: Mutable state for the current training run.
        phase: Active ``"decoder"`` or ``"diffusion"`` phase.
        clean: Clean latent batch.
        target: Target values or tensor aligned with the current rows.
        torch_generator: Torch random stream for tensor sampling.
        numpy_generator: NumPy random stream for masking-mode selection.
    """
    # Evaluate both phases through the same noisy denoising path as training.
    batch = _build_loss_batch(context, clean, target, torch_generator, numpy_generator)
    return _phase_loss(context, phase, batch)


def _sample_observation_mask(
    rows: int,
    tokens: int,
    *,
    has_target: bool,
    probabilities: dict[str, float],
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample observed-token masks from configured training modes.

    Args:
        rows: Number of rows to sample or process.
        tokens: Number of latent tokens per row.
        has_target: Whether the fitted latent layout includes a target token.
        probabilities: Normalized probabilities for supported masking modes.
        rng: Torch random stream used for tensor sampling.
        device: Torch device used by tensors and neural components.
    """
    modes = tuple(probabilities)
    mode = rng.choice(modes, p=np.asarray([probabilities[key] for key in modes]))
    # Internal polarity is ``True means observed``; unconditional mode has no clamps.
    result = np.zeros((rows, tokens), dtype=bool)
    if mode == "target" and has_target:
        result[:, :] = True
        result[:, -1] = False
    elif mode == "random":
        result = rng.random((rows, tokens)) > 0.25
        if has_target:
            result[:, -1] = True
        # Every conditioned row retains at least one token to regenerate.
        fully_observed = np.flatnonzero(result.all(axis=1))
        result[
            fully_observed,
            rng.integers(0, tokens - int(has_target), len(fully_observed)),
        ] = False
    return torch.as_tensor(result, dtype=torch.bool, device=device)
