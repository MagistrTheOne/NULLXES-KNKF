"""Export Hugging Face tokenizer metadata bundle for KNK-VF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_tokenizer_config(config: dict) -> dict:
    symbols = config["user_defined_symbols"]
    return {
        "tokenizer_class": config["hf_export"]["tokenizer_class"],
        "model_max_length": config["hf_export"]["model_max_length"],
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<unk>",
        "added_tokens_decoder": {
            str(128000 + idx): {"content": token, "lstrip": False, "rstrip": False, "special": True}
            for idx, token in enumerate(symbols)
        },
        "additional_special_tokens": symbols,
        "clean_up_tokenization_spaces": False,
        "use_default_system_prompt": False,
        "chat_template": (Path(__file__).parent / "templates" / "knk_vf_chat_v2.jinja").read_text(
            encoding="utf-8"
        ),
    }


def build_special_tokens_map() -> dict:
    return {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<unk>",
        "additional_special_tokens": [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|tool|>",
            "<|code|>",
            "<|/code|>",
            "<|think|>",
            "<|/think|>",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer/knk_vf_tokenizer_128k.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tokenizer_config.json").write_text(
        json.dumps(build_tokenizer_config(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output / "special_tokens_map.json").write_text(
        json.dumps(build_special_tokens_map(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    template_src = Path(__file__).parent / "templates" / "knk_vf_chat_v2.jinja"
    (args.output / "chat_template.jinja").write_text(template_src.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": ["tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]}, indent=2))


if __name__ == "__main__":
    main()
