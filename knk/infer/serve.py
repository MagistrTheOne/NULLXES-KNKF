"""Serving configuration helpers for KNK-VF."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_vllm_command(config_path: Path, model_path: str) -> list[str]:
    config = load_yaml(config_path)
    runtime = config["runtime"]
    command = [
        "vllm",
        "serve",
        model_path,
        "--max-model-len",
        str(runtime["max_model_len"]),
        "--tensor-parallel-size",
        str(runtime["tensor_parallel_size"]),
    ]
    if runtime.get("enable_prefix_caching"):
        command.append("--enable-prefix-caching")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    command = build_vllm_command(args.config, args.model_path)
    print(" ".join(shlex.quote(part) for part in command))


if __name__ == "__main__":
    main()
