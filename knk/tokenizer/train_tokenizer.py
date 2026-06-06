"""Train a multilingual SentencePiece tokenizer for KNK-VF."""

from __future__ import annotations

import argparse
from pathlib import Path
import unicodedata


SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_corpus(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            target.write(normalize_text(line))


def train_sentencepiece(
    input_path: Path,
    model_prefix: Path,
    vocab_size: int = 128000,
    byte_fallback: bool = True,
) -> None:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install sentencepiece to train the tokenizer.") from exc

    user_defined_symbols = ",".join(SPECIAL_TOKENS)
    spm.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        normalization_rule_name="nfkc",
        byte_fallback=byte_fallback,
        user_defined_symbols=user_defined_symbols,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--vocab-size", default=128000, type=int)
    parser.add_argument("--normalized-output", type=Path)
    args = parser.parse_args()

    normalized = args.normalized_output or args.output_prefix.with_suffix(".normalized.txt")
    normalize_corpus(args.input, normalized)
    train_sentencepiece(normalized, args.output_prefix, args.vocab_size)


if __name__ == "__main__":
    main()
