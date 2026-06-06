"""Tier-weighted sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class DataSource:
    name: str
    weight: float
    path: str | None = None


def normalize_weights(sources: Sequence[DataSource]) -> list[float]:
    total = sum(source.weight for source in sources)
    if total <= 0:
        raise ValueError("At least one source must have positive weight")
    return [source.weight / total for source in sources]


def sample_source(sources: Sequence[DataSource], rng: random.Random | None = None) -> DataSource:
    rng = rng or random
    weights = normalize_weights(sources)
    return rng.choices(list(sources), weights=weights, k=1)[0]
