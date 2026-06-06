"""Optimizer and scheduler helpers for proxy training."""

from __future__ import annotations

import math

import torch


def build_adamw(parameters, lr: float, weight_decay: float = 0.1) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=weight_decay,
    )


def wsd_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float = 0.02) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    stable_steps = max(warmup_steps + 1, int(total_steps * 0.8))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step < stable_steps:
        return base_lr
    progress = (step - stable_steps) / max(total_steps - stable_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
