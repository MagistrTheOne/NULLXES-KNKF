"""Training metric aggregation and alert checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    loss: float
    grad_norm: float
    tokens_per_second: float
    mfu: float | None = None
    router_entropy: float | None = None
    max_expert_share: float | None = None
    skipped_tokens_ratio: float = 0.0


def alert_reasons(metrics: TrainingMetrics, running_loss_median: float | None = None) -> list[str]:
    reasons: list[str] = []
    if running_loss_median and metrics.loss > 2.0 * running_loss_median:
        reasons.append("loss_spike")
    if metrics.grad_norm != metrics.grad_norm or metrics.grad_norm > 10_000:
        reasons.append("grad_norm_unstable")
    if metrics.max_expert_share is not None and metrics.max_expert_share > 0.40:
        reasons.append("expert_collapse")
    if metrics.skipped_tokens_ratio > 0.05:
        reasons.append("skipped_tokens_high")
    return reasons
