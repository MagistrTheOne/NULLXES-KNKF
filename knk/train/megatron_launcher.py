"""Megatron-Core launch command builder for KNK-VF target pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_megatron_command(config_path: Path) -> list[str]:
    config = load_yaml(config_path)
    parallel = config["parallelism"]
    command = [
        "torchrun",
        "--nproc_per_node",
        str(config["hardware"]["gpus_per_node"]),
        "-m",
        "megatron.training.pretrain_gpt",
        "--tensor-model-parallel-size",
        str(parallel["tensor_parallel"]),
        "--pipeline-model-parallel-size",
        str(parallel["pipeline_parallel"]),
        "--expert-model-parallel-size",
        str(parallel["expert_parallel"]),
        "--context-parallel-size",
        str(parallel["context_parallel"]),
        "--use-distributed-optimizer",
        "--sequence-parallel",
        "--bf16",
        "--attention-backend",
        "flash",
        "--save",
        "checkpoints/target",
        "--load",
        "checkpoints/target",
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--print-only", action="store_true", default=True)
    args = parser.parse_args()
    command = build_megatron_command(args.config)
    print(" ".join(shlex.quote(part) for part in command))


if __name__ == "__main__":
    main()
