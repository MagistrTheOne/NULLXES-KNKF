"""Minimal muP calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MuPScalePoint:
    width_multiplier: float
    learning_rate: float
    loss: float


def transfer_learning_rate(points: list[MuPScalePoint], target_width_multiplier: float) -> float:
    """Fit the simple LR ~= c / width rule used for first-pass proxy transfer."""
    if not points:
        raise ValueError("At least one muP scale point is required")
    constants = [point.learning_rate * point.width_multiplier for point in points]
    c = sum(constants) / len(constants)
    return c / target_width_multiplier
