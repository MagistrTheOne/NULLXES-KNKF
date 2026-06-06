"""Streaming text loaders for proxy training and evaluation dry-runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def iter_jsonl_text(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield row["text"] if isinstance(row, dict) else str(row)


def pack_token_ids(token_ids: list[int], sequence_length: int) -> Iterator[list[int]]:
    for start in range(0, len(token_ids) - sequence_length + 1, sequence_length):
        yield token_ids[start : start + sequence_length]
