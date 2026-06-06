"""Deterministic weight initialization policies for KNK-VF checkpoints."""

from __future__ import annotations

import hashlib
from enum import Enum

import torch


class InitKind(str, Enum):
    RMSNORM = "rmsnorm"
    EMBEDDING = "embedding"
    LINEAR_STD = "linear_std"
    LINEAR_RESIDUAL = "linear_residual"
    ROUTER = "router"


def tensor_seed(name: str, global_seed: int = 42) -> int:
    digest = hashlib.sha256(f"{global_seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def init_tensor(
    name: str,
    shape: tuple[int, ...],
    kind: InitKind,
    *,
    num_layers: int,
    base_std: float = 0.02,
    router_std: float = 0.001,
    global_seed: int = 42,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kind is InitKind.RMSNORM:
        return torch.ones(shape, device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(tensor_seed(name, global_seed))

    if kind is InitKind.ROUTER:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype) * router_std

    std = base_std
    if kind is InitKind.LINEAR_RESIDUAL:
        std = base_std / (2 * num_layers) ** 0.5

    return torch.randn(shape, generator=generator, device=device, dtype=dtype) * std
