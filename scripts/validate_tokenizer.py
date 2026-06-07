"""Validate NULLXES KNK-VF tokenizer on EN/RU/code holdout slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import yaml


SPECIALS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|code|>",
    "<|/code|>",
    "<|think|>",
    "<|/think|>",
]


def load_holdout(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for message in row.get("messages", []):
            texts.append(message["content"])
    return texts


def chars_per_token(sp: spm.SentencePieceProcessor, texts: list[str]) -> float:
    chars = sum(len(t) for t in texts)
    toks = sum(len(sp.encode(t, out_type=int)) for t in texts)
    return chars / max(toks, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/tokenizer/knk_vf_tokenizer_128k.yaml"), type=Path)
    parser.add_argument("--holdout", default=Path("data/bootstrap"), type=Path)
    args = parser.parse_args()

    tok_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = tok_cfg["validation_targets"]
    sp = spm.SentencePieceProcessor(model_file=str(args.model))

    texts = []
    for path in sorted(args.holdout.glob("*.jsonl")):
        texts.extend(load_holdout(path))
    if not texts:
        raise SystemExit(f"No holdout text found under {args.holdout}")

    unk_id = sp.unk_id()
    unk_hits = sum(1 for t in texts for tid in sp.encode(t, out_type=int) if tid == unk_id)
    total_tokens = sum(len(sp.encode(t, out_type=int)) for t in texts)
    unk_rate = unk_hits / max(total_tokens, 1)
    cpt = chars_per_token(sp, texts)

    en = [t for t in texts if any(ord(c) < 128 for c in t) and not any("\u0400" <= c <= "\u04FF" for c in t)]
    ru = [t for t in texts if any("\u0400" <= c <= "\u04FF" for c in t)]
    code = [t for t in texts if "def " in t or "import " in t or "<|code|>" in t or "yaml" in t.lower()]

    special_failures = []
    for token in SPECIALS:
        pieces = sp.encode(token, out_type=str)
        if pieces != [token]:
            special_failures.append({"token": token, "pieces": pieces})

    report = {
        "unk_rate": unk_rate,
        "chars_per_token_all": cpt,
        "chars_per_token_en": chars_per_token(sp, en) if en else None,
        "chars_per_token_ru": chars_per_token(sp, ru) if ru else None,
        "chars_per_token_code": chars_per_token(sp, code) if code else None,
        "special_token_failures": special_failures,
        "targets": targets,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failures = []
    if unk_rate > targets["unk_rate_max"]:
        failures.append(f"unk_rate {unk_rate:.6f} > {targets['unk_rate_max']}")
    if en and report["chars_per_token_en"] < targets["en_holdout_chars_per_token_min"]:
        failures.append("en chars/token too low")
    if ru and report["chars_per_token_ru"] < targets["ru_holdout_chars_per_token_min"]:
        failures.append("ru chars/token too low")
    if code and report["chars_per_token_code"] < targets["code_holdout_chars_per_token_min"]:
        failures.append("code chars/token too low")
    if special_failures:
        failures.append(f"{len(special_failures)} special tokens split incorrectly")

    if failures:
        raise SystemExit("Tokenizer validation failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
