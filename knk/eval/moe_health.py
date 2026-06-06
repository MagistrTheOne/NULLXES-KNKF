"""MoE health metrics for routing diagnostics."""

from __future__ import annotations

import torch


def expert_entropy(utilization: torch.Tensor) -> torch.Tensor:
    probs = utilization.float() / utilization.sum().clamp_min(1.0e-9)
    return -(probs * probs.clamp_min(1.0e-9).log()).sum()


def has_expert_collapse(utilization: torch.Tensor, max_share: float = 0.40) -> bool:
    if utilization.numel() == 0:
        return False
    probs = utilization.float() / utilization.sum().clamp_min(1.0e-9)
    return bool(probs.max().item() > max_share)
