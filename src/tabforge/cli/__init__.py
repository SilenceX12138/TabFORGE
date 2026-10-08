"""Command-line orchestration for the public TabFORGE estimator APIs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from tabforge import (
    TabFORGEArchitectureConfig,
    TabFORGEClassifier,
    TabFORGEDiffusionConfig,
    TabFORGEEmbedder,
    TabFORGEEmbeddingConfig,
    TabFORGEGenerator,
    TabFORGEImputer,
    TabFORGERegressor,
    TabFORGERuntimeConfig,
    TabFORGETrainingConfig,
    __version__,
    restore_from_checkpoint,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected TabFORGE command.

    Args:
        argv: Optional command-line tokens; ``None`` reads ``sys.argv``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    rank, world_size, owns_process_group = _start_process_group(args)
    try:
        if args.command == "fit":
            _fit(args, rank=rank, world_size=world_size)
        elif args.command == "generate":
            _generate(args, rank=rank)
        elif args.command == "predict":
            _predict(args, rank=rank)
        elif args.command == "embed":
            _embed(args, rank=rank)
        elif args.command == "impute":
            _impute(args, rank=rank)
        return 0
    finally:
        if owns_process_group:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()


def _build_parser() -> argparse.ArgumentParser:
    """Build the root parser and command-specific argument groups."""
    parser = argparse.ArgumentParser(prog="tabforge", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    _add_fit_command(commands)
    _add_generate_command(commands)
    _add_predict_command(commands)
    _add_embed_command(commands)
    _add_impute_command(commands)
    return parser


def _add_fit_command(commands: argparse._SubParsersAction) -> None:
    """Register estimator fitting arguments.

    Args:
        commands: Argparse subparser collection receiving a command.
    """
    fit = commands.add_parser("fit", help="fit an estimator from a CSV file")
    fit.add_argument("--input", required=True, type=Path)
    fit.add_argument("--checkpoint", required=True, type=Path)
    fit.add_argument(
        "--estimator",
        choices=("generator", "classifier", "regressor", "embedder", "imputer"),
        default="generator",
    )
    fit.add_argument(
        "--task",
        choices=("classification", "regression", "unsupervision"),
        help="required when fitting a generator or embedder",
    )
    fit.add_argument("--target", help="target column for supervised estimators")
    fit.add_argument(
        "--categorical-features",
        default="",
        help="comma-separated column names or integer positions",
    )
    fit.add_argument("--model-path", default="auto", help="auto, mock, or a local TabPFN checkpoint")
    fit.add_argument("--embedding-dimension", type=int)
    fit.add_argument("--max-steps", type=int, default=10_000)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--strategy", choices=("auto", "single_device", "ddp"), default="auto")
    fit.add_argument("--gradient-accumulation", type=int, default=1)
    fit.add_argument("--deterministic", action="store_true")
    fit.add_argument("--log-wandb", action="store_true")
    fit.add_argument("--wandb-project", default="tabforge")
    fit.add_argument("--wandb-entity")
    fit.add_argument("--wandb-dir", default="./logs/wandb")
    fit.add_argument("--random-state", type=int)
    fit.add_argument("--n-prediction-samples", type=int, default=32)
    fit.add_argument("--n-imputation-samples", type=int, default=1)


def _add_generate_command(commands: argparse._SubParsersAction) -> None:
    """Register unconditional generation arguments.

    Args:
        commands: Argparse subparser collection receiving a command.
    """
    generate = commands.add_parser("generate", help="generate rows from a fitted generator checkpoint")
    generate.add_argument("--checkpoint", required=True, type=Path)
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--n-samples", required=True, type=int)
    generate.add_argument("--random-state", type=int)
    generate.add_argument("--device", default="auto")


def _add_predict_command(commands: argparse._SubParsersAction) -> None:
    """Register supervised prediction arguments.

    Args:
        commands: Argparse subparser collection receiving a command.
    """
    predict = commands.add_parser("predict", help="run a fitted classifier or regressor")
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--random-state", type=int)
    predict.add_argument("--device", default="auto")


def _add_embed_command(commands: argparse._SubParsersAction) -> None:
    """Register feature embedding arguments.

    Args:
        commands: Argparse subparser collection receiving a command.
    """
    embed = commands.add_parser("embed", help="write fitted full-grid embeddings as .npy")
    embed.add_argument("--checkpoint", required=True, type=Path)
    embed.add_argument("--input", required=True, type=Path)
    embed.add_argument("--output", required=True, type=Path)
    embed.add_argument("--device", default="auto")


def _add_impute_command(commands: argparse._SubParsersAction) -> None:
    """Register feature imputation arguments.

    Args:
        commands: Argparse subparser collection receiving a command.
    """
    impute = commands.add_parser("impute", help="impute a CSV with a fitted imputer")
    impute.add_argument("--checkpoint", required=True, type=Path)
    impute.add_argument("--input", required=True, type=Path)
    impute.add_argument("--output", required=True, type=Path)
    impute.add_argument("--random-state", type=int)
    impute.add_argument("--device", default="auto")


def _fit(args: argparse.Namespace, *, rank: int, world_size: int) -> None:
    """Fit and save one estimator checkpoint.

    Args:
        args: Parsed command-line argument namespace.
        rank: Current distributed rank.
        world_size: Number of participating distributed ranks.
    """
    # === Load training features and target ===
    data, y = _load_fit_data(args)
    categorical = _parse_categorical_features(args.categorical_features)
    estimator_cls = _fit_estimator_class(args.estimator, y)
    # === Build public estimator configuration ===
    kwargs = _fit_estimator_kwargs(args, categorical, world_size)
    kwargs.update(_task_specific_fit_kwargs(args))
    model = estimator_cls(**kwargs).fit(data, y)
    _barrier()
    if rank == 0:
        model.save_checkpoint(args.checkpoint)


def _load_fit_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load a CSV and separate its optional target column.

    Args:
        args: Parsed command-line argument namespace.
    """
    data = pd.read_csv(args.input)
    if args.target is None:
        return data, None
    if args.target not in data.columns:
        raise ValueError(f"Target column {args.target!r} is not present in {args.input}")
    return data, data.pop(args.target)


def _fit_estimator_class(estimator_type: str, y: pd.Series | None) -> type:
    """Resolve the requested estimator and its target contract.

    Args:
        estimator_type: Serialized or CLI estimator type identifier.
        y: Target values aligned with the feature rows.
    """
    estimator_cls = {
        "generator": TabFORGEGenerator,
        "classifier": TabFORGEClassifier,
        "regressor": TabFORGERegressor,
        "embedder": TabFORGEEmbedder,
        "imputer": TabFORGEImputer,
    }[estimator_type]
    if estimator_type in {"classifier", "regressor"} and y is None:
        raise ValueError(f"{estimator_type} fitting requires --target")
    if estimator_type == "imputer" and y is not None:
        raise ValueError("imputer fitting does not accept --target")
    return estimator_cls


def _fit_estimator_kwargs(args: argparse.Namespace, categorical: tuple, world_size: int) -> dict[str, Any]:
    """Build configuration shared by all CLI estimators.

    Args:
        args: Parsed command-line argument namespace.
        categorical: Categorical feature names or integer positions.
        world_size: Number of participating distributed ranks.
    """
    return {
        "categorical_features": categorical,
        "embedding_config": TabFORGEEmbeddingConfig(
            model_path=args.model_path,
            n_folds=0 if args.model_path in {"mock", "synthetic", "none"} else 10,
        ),
        "architecture_config": TabFORGEArchitectureConfig(embedding_dimension=args.embedding_dimension),
        "diffusion_config": TabFORGEDiffusionConfig(),
        "training_config": TabFORGETrainingConfig(max_steps=args.max_steps, batch_size=args.batch_size),
        "runtime_config": _runtime_config(args, world_size=world_size),
        "random_state": args.random_state,
    }


def _task_specific_fit_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build task and inference-specific estimator configuration.

    Args:
        args: Parsed command-line argument namespace.
    """
    estimator_kwargs = {}
    estimator_type = args.estimator
    if estimator_type in {"generator", "embedder"}:
        if args.task is None:
            raise ValueError(f"{estimator_type} fitting requires --task")
        estimator_kwargs["task"] = args.task
    if estimator_type == "classifier":
        estimator_kwargs["n_prediction_samples"] = args.n_prediction_samples
    elif estimator_type == "regressor":
        estimator_kwargs["n_prediction_samples"] = args.n_prediction_samples
    elif estimator_type == "imputer":
        estimator_kwargs["n_imputation_samples"] = args.n_imputation_samples
    return estimator_kwargs


def _generate(args: argparse.Namespace, *, rank: int) -> None:
    """Generate synthetic rows through the distributed estimator API.

    Args:
        args: Parsed command-line argument namespace.
        rank: Current distributed rank.
    """
    model = restore_from_checkpoint(args.checkpoint, device=args.device)
    generator = _require_type(model, TabFORGEGenerator)
    generated = generator.generate(args.n_samples, random_state=args.random_state)
    if rank == 0:
        _write_generated(generated, args.output)


def _predict(args: argparse.Namespace, *, rank: int) -> None:
    """Run classifier or regressor inference through the estimator API.

    Args:
        args: Parsed command-line argument namespace.
        rank: Current distributed rank.
    """
    model = restore_from_checkpoint(args.checkpoint, device=args.device)
    data = pd.read_csv(args.input)
    if isinstance(model, TabFORGEClassifier):
        probabilities = model.predict_proba(data, random_state=args.random_state)
        result = pd.DataFrame(probabilities, columns=[f"proba_{value}" for value in model.classes_])
    elif isinstance(model, TabFORGERegressor):
        result = pd.DataFrame({"prediction": model.predict(data, random_state=args.random_state)})
    else:
        raise TypeError("predict requires a classifier or regressor checkpoint")
    if rank == 0:
        result.to_csv(args.output, index=False)


def _embed(args: argparse.Namespace, *, rank: int) -> None:
    """Extract full-grid embeddings through the distributed estimator API.

    Args:
        args: Parsed command-line argument namespace.
        rank: Current distributed rank.
    """
    model = restore_from_checkpoint(args.checkpoint, device=args.device)
    embedder = _require_type(model, TabFORGEEmbedder)
    data = pd.read_csv(args.input)
    result = embedder.transform(data)
    if rank == 0:
        np.save(args.output, result)


def _impute(args: argparse.Namespace, *, rank: int) -> None:
    """Impute rows through the distributed estimator API.

    Args:
        args: Parsed command-line argument namespace.
        rank: Current distributed rank.
    """
    model = restore_from_checkpoint(args.checkpoint, device=args.device)
    imputer = _require_type(model, TabFORGEImputer)
    data = pd.read_csv(args.input)
    result = imputer.transform(data, random_state=args.random_state)
    if rank == 0:
        result.to_csv(args.output, index=False)


def _runtime_config(args: argparse.Namespace, *, world_size: int) -> TabFORGERuntimeConfig:
    """Translate CLI device settings into runtime configuration.

    Args:
        args: Parsed command-line argument namespace.
        world_size: Number of participating distributed ranks.
    """
    distributed = world_size > 1
    strategy = "ddp" if distributed and args.strategy == "auto" else args.strategy
    return TabFORGERuntimeConfig(
        device=args.device,
        strategy=strategy,
        gradient_accumulation=args.gradient_accumulation,
        deterministic=args.deterministic,
        log_wandb=args.log_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_dir=args.wandb_dir,
    )


def _start_process_group(args: argparse.Namespace) -> tuple[int, int, bool]:
    """Join an externally launched distributed process group when required.

    Args:
        args: Parsed command-line argument namespace.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1 or torch.distributed.is_initialized():
        return rank, world_size, False
    use_cuda = torch.cuda.is_available() and (
        getattr(args, "device", "auto").startswith("cuda") or getattr(args, "device", "auto") == "auto"
    )
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
    backend = "nccl" if use_cuda else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, True


def _barrier() -> None:
    """Synchronize ranks when distributed execution is active."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _parse_categorical_features(value: str) -> tuple[int | str, ...] | None:
    """Parse comma-separated categorical names or integer positions.

    Args:
        value: Object or tensor handled by the helper.
    """
    if not value:
        return None
    result: list[int | str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item) if item.lstrip("-").isdigit() else item)
    return tuple(result)


def _require_type(model: object, expected: type) -> Any:
    """Require a checkpoint estimator type for a command.

    Args:
        model: External embedding model or fitted estimator instance.
        expected: Required estimator class for the current command.
    """
    if not isinstance(model, expected):
        raise TypeError(f"Expected a {expected.__name__} checkpoint, got {type(model).__name__}")
    return model


def _write_generated(value: Any, path: Path) -> None:
    """Write generated features and an optional target to CSV.

    Args:
        value: Object or tensor handled by the helper.
        path: Checkpoint source or destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, tuple):
        features, target = value
        output = features.copy()
        output["target"] = target
    else:
        output = value
    output.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    raise SystemExit(main())
