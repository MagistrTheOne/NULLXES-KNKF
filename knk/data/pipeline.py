"""Corpus preparation pipeline: normalize, deduplicate, filter, scrub, and shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def quality_score(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(char.isalpha() for char in text)
    printable = sum(char.isprintable() for char in text)
    return min(1.0, (alpha / max(len(text), 1)) * 1.5 + (printable / max(len(text), 1)) * 0.2)


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_jsonl(input_path: Path, output_path: Path, min_quality_score: float = 0.55) -> dict:
    seen: set[str] = set()
    stats = {"read": 0, "written": 0, "deduped": 0, "filtered": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            stats["read"] += 1
            text = normalize(line)
            digest = stable_hash(text)
            if digest in seen:
                stats["deduped"] += 1
                continue
            seen.add(digest)
            if quality_score(text) < min_quality_score:
                stats["filtered"] += 1
                continue
            target.write(json.dumps({"text": scrub_pii(text), "sha256": digest}) + "\n")
            stats["written"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-quality-score", default=0.55, type=float)
    args = parser.parse_args()
    stats = prepare_jsonl(args.input, args.output, args.min_quality_score)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
