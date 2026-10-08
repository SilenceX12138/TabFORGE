"""Small torchrun helpers shared by the estimator API and CLI."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import distributed as dist


def initialize(device: str = "auto") -> bool:
    """Join a torchrun process group and select this worker's CUDA device.

    Args:
        device: Runtime device selector from the public configuration.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or dist.is_initialized():
        return False
    use_cuda = torch.cuda.is_available() and (device == "auto" or str(device).startswith("cuda"))
    if use_cuda:
        torch.cuda.set_device(local_rank())
    dist.init_process_group(backend="nccl" if use_cuda else "gloo", init_method="env://")
    return True


def active() -> bool:
    """Return whether a multi-rank process group is active."""
    return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)


def rank() -> int:
    """Return the active global rank, defaulting to zero."""
    return dist.get_rank() if active() else 0


def world_size() -> int:
    """Return the active world size, defaulting to one."""
    return dist.get_world_size() if active() else 1


def local_rank() -> int:
    """Return the torchrun local rank."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def shard_indices(length: int) -> np.ndarray:
    """Assign disjoint interleaved row indices to this rank.

    Args:
        length: Total number of input rows.
    """
    return np.arange(rank(), length, world_size(), dtype=np.int64)


def split_count(total: int) -> list[int]:
    """Split an item count evenly across active ranks.

    Args:
        total: Total number of items to distribute.
    """
    base, remainder = divmod(total, world_size())
    return [base + int(item < remainder) for item in range(world_size())]


def gather_objects(value: Any) -> list[Any]:
    """Gather one Python object from every rank.

    Args:
        value: Current rank's serializable object.
    """
    if not active():
        return [value]
    gathered = [None] * world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def local_cuda_device(device: str) -> str:
    """Map a CUDA request onto this torchrun worker's local device.

    Args:
        device: Public device selector.
    """
    if active() and torch.cuda.is_available() and (device == "auto" or str(device).startswith("cuda")):
        return f"cuda:{local_rank()}"
    return device
