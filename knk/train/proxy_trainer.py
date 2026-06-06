"""FSDP proxy trainer entrypoint for H200 validation runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from knk.model.knk_vf import KNKVFConfig, KNKVFForCausalLM
from knk.train.optimizer import build_adamw


def assert_cluster_execution() -> None:
    """Prevent accidental workstation runs; KNKF execution is H200/B300-only."""
    allowed = os.environ.get("KNKF_CLUSTER_EXECUTION") == "1"
    accelerator = os.environ.get("KNKF_ACCELERATOR", "").lower()
    if not allowed or accelerator not in {"h200", "b300"}:
        raise RuntimeError(
            "Local KNKF execution is forbidden. Set KNKF_CLUSTER_EXECUTION=1 and "
            "KNKF_ACCELERATOR=h200 or b300 only inside approved cluster jobs."
        )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_from_config(model_config_path: Path) -> KNKVFForCausalLM:
    config_data = load_yaml(model_config_path)
    config = KNKVFConfig.from_model_dict(config_data["model"])
    return KNKVFForCausalLM(config)


def dry_run(model_config_path: Path, sequence_length: int = 16) -> dict[str, float]:
    """Run a tiny forward/backward pass using an intentionally truncated proxy config."""
    data = load_yaml(model_config_path)
    model_data = data["model"]
    model_data = {
        **model_data,
        "vocab_size": min(512, int(model_data["vocab_size"])),
        "max_position_embeddings": min(sequence_length, int(model_data["max_position_embeddings"])),
        "num_layers": min(2, int(model_data["num_layers"])),
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "dense_prefix_layers": 1,
        "ffn_hidden_size": 256,
        "expert_ffn_hidden_size": 256,
        "num_routed_experts": 4,
        "routed_top_k": 2,
    }
    model = KNKVFForCausalLM(KNKVFConfig.from_model_dict(model_data))
    optimizer = build_adamw(model.parameters(), lr=1.0e-4)
    input_ids = torch.randint(0, model.config.vocab_size, (2, sequence_length))
    output = model(input_ids, labels=input_ids)
    loss = output["loss"]
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    assert_cluster_execution()
    train_config = load_yaml(args.config)
    model_config_path = Path(train_config["model_config"])
    if args.dry_run:
        print(dry_run(model_config_path))
        return
    raise RuntimeError(
        "Full proxy training must be launched with torchrun on H200 hardware. "
        "Use --dry-run for local validation."
    )


if __name__ == "__main__":
    main()
