"""Prepare a local KNK-VF checkpoint directory for LLaMA-Factory/Transformers.

The script is intentionally local-only: it never downloads a foreign base model.
It copies KNK-VF remote code into the initialized checkpoint and writes tokenizer
metadata around the NULLXES SentencePiece model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SPECIAL_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|code|>",
    "<|/code|>",
    "<|think|>",
    "<|/think|>",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_sentencepiece(model_dir: Path, tokenizer_model: Path | None) -> Path:
    candidates = [
        tokenizer_model,
        model_dir / "tokenizer.model",
        model_dir / "tokenizer" / "knk_vf.model",
        Path("/workspace/artifacts/knk_vf_tokenizer_lab/knk_vf.model"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "SentencePiece model not found. Pass --tokenizer-model or copy knk_vf.model into the checkpoint."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/checkpoints/knk_vf_lab_38b"))
    parser.add_argument("--tokenizer-model", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")

    config_path = model_dir / "config.json"
    config = read_json(config_path)
    config.update(
        {
            "architectures": ["KNKVFForCausalLM"],
            "model_type": "knk_vf",
            "auto_map": {
                "AutoConfig": "configuration_knk_vf.KNKVFConfig",
                "AutoModelForCausalLM": "modeling_knk_vf.KNKVFForCausalLM",
            },
            "trust_remote_code": True,
        }
    )
    write_json(config_path, config)

    hf_dir = args.repo_root / "knk" / "hf"
    for filename in ["configuration_knk_vf.py", "modeling_knk_vf.py"]:
        shutil.copy2(hf_dir / filename, model_dir / filename)

    sp_model = find_sentencepiece(model_dir, args.tokenizer_model)
    shutil.copy2(sp_model, model_dir / "tokenizer.model")

    template_path = args.repo_root / "knk" / "tokenizer" / "templates" / "knk_vf_chat_v2.jinja"
    chat_template = template_path.read_text(encoding="utf-8")
    (model_dir / "chat_template.jinja").write_text(chat_template, encoding="utf-8")

    write_json(
        model_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "LlamaTokenizer",
            "model_max_length": int(config.get("max_position_embeddings", 131072)),
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "pad_token": "<|pad|>",
            "unk_token": "<unk>",
            "additional_special_tokens": SPECIAL_TOKENS,
            "clean_up_tokenization_spaces": False,
            "legacy": False,
            "chat_template": chat_template,
        },
    )
    write_json(
        model_dir / "special_tokens_map.json",
        {
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "pad_token": "<|pad|>",
            "unk_token": "<unk>",
            "additional_special_tokens": SPECIAL_TOKENS,
        },
    )

    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "base_model": "KNK-VF local initialized checkpoint",
                "foreign_model_downloads": False,
                "remote_code": ["configuration_knk_vf.py", "modeling_knk_vf.py"],
                "tokenizer_model": str(model_dir / "tokenizer.model"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

