"""Hybrid local/global attention pattern helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridAttentionPattern:
    local_window: int
    global_every: int = 4

    def local_window_for_layer(self, layer_idx: int) -> int | None:
        """Return None for full attention layers, otherwise the sliding window size."""
        if self.global_every <= 0:
            return self.local_window
        is_global = (layer_idx + 1) % self.global_every == 0
        return None if is_global else self.local_window
