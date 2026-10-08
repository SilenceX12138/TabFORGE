"""Optional W&B tracing for the standalone trainer."""

from __future__ import annotations

from typing import Any

from torch import distributed as dist


class TrainingLogger:
    """Log training traces to an existing or standalone W&B run.

    ``context`` may be a W&B run or a Lightning ``WandbLogger``.  Passing an
    existing context never creates another run, which keeps an integration
    caller and TabFORGE in one experiment.
    """

    def __init__(self, runtime_config: Any, context: Any = None) -> None:
        """Reuse a supplied logger or create the configured standalone W&B run.

        Args:
            runtime_config: Resolved device, distribution, and logging configuration.
            context: Mutable state for the current training run.
        """

        if _distributed_non_primary():
            self.run = None
            self._owns_run = False
            return
        self.run = _as_run(context)
        self._owns_run = False
        if self.run is None and runtime_config.log_wandb:
            import wandb

            self.run = wandb.run
            if self.run is None:
                self.run = wandb.init(
                    project=runtime_config.wandb_project,
                    entity=runtime_config.wandb_entity,
                    dir=runtime_config.wandb_dir,
                )
                self._owns_run = True
        if self.run is not None and hasattr(self.run, "define_metric"):
            self.run.define_metric("training/optimizer_step")
            for pattern in ("training/*", "valid/*"):
                self.run.define_metric(pattern, step_metric="training/optimizer_step")

    @property
    def enabled(self) -> bool:
        """Return whether trace calls have a logging destination."""
        return self.run is not None

    def log(self, values: dict[str, Any], *, step: int) -> None:
        """Write one optimizer-step trace when logging is enabled.

        Args:
            values: Values handled by the current transformation.
            step: Completed global optimizer-step index.
        """
        if self.run is not None:
            if hasattr(self.run, "define_metric"):
                self.run.log(values)
            else:
                self.run.log(values, step=step)

    def finish(self) -> None:
        """Finish only runs created by this trainer."""
        if self._owns_run and self.run is not None:
            self.run.finish()


def _as_run(context: Any) -> Any:
    """Extract a W&B run from a run object or Lightning logger.

    Args:
        context: Mutable state for the current training run.
    """
    if context is None:
        return None
    if hasattr(context, "log") and hasattr(context, "summary"):
        return context
    experiment = getattr(context, "experiment", None)
    if experiment is not None and hasattr(experiment, "log"):
        return experiment
    return None


def _distributed_non_primary() -> bool:
    """Keep trace output on rank zero when the trainer is distributed."""
    return bool(dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
