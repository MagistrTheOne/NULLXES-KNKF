"""B300/H200 sharded checkpoint initialization for KNK-VF target models."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from knk.model.init_weights import InitKind, init_tensor
from knk.model.param_counter import count_parameters, load_config, validate_targets
from knk.model.shard_manifest import build_manifest, group_into_shards


def assert_cluster_execution() -> None:
    allowed = os.environ.get("KNKF_CLUSTER_EXECUTION") == "1"
    accelerator = os.environ.get("KNKF_ACCELERATOR", "").lower()
    if not allowed or accelerator not in {"h200", "b300"}:
        raise RuntimeError(
            "KNKF init is cluster-only. Set KNKF_CLUSTER_EXECUTION=1 and "
            "KNKF_ACCELERATOR=b300 (or h200) before running."
        )


def build_hf_config(model: dict, report_total: int, report_active: int) -> dict:
    return {
        "architectures": ["KNKVFForCausalLM"],
        "model_type": "knk_vf",
        "vocab_size": model["vocab_size"],
        "hidden_size": model["hidden_size"],
        "num_hidden_layers": model["num_layers"],
        "num_attention_heads": model["num_attention_heads"],
        "num_key_value_heads": model["num_key_value_heads"],
        "head_dim": model["head_dim"],
        "intermediate_size": model["ffn_hidden_size"],
        "expert_intermediate_size": model["expert_ffn_hidden_size"],
        "num_routed_experts": model["num_routed_experts"],
        "num_shared_experts": model.get("num_shared_experts", 1),
        "routed_top_k": model["routed_top_k"],
        "dense_prefix_layers": model["dense_prefix_layers"],
        "max_position_embeddings": model["max_position_embeddings"],
        "rope_theta": model["rope_theta"],
        "rms_norm_eps": model.get("norm_eps", 1e-6),
        "tie_word_embeddings": model.get("tie_embeddings", False),
        "torch_dtype": "bfloat16",
        "total_params": report_total,
        "active_params": report_active,
        "init_policy": "llama_neox_residual",
        "owner": "NULLXES",
        "contact": "ceo@nullxes.com",
    }


def write_model_card(output_dir: Path, model_name: str, report_total: int, report_active: int) -> None:
    card = f"""---
library_name: transformers
tags:
- nullxes
- knkf
- moe
- text-generation
license: other
---

# {model_name}

NULLXES KNK-VF target checkpoint initialized with deterministic sharded random weights.

- Total parameters: `{report_total:,}`
- Active parameters: `{report_active:,}`
- Initialization only. Not pretrained.
- Contact: ceo@nullxes.com
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def upload_to_hf(output_dir: Path, repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for --upload")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initialize KNK-VF 1T sharded checkpoint (bf16)",
    )


def initialize_checkpoint(
    config_path: Path,
    output_dir: Path,
    *,
    max_shard_gb: float = 4.0,
    device: str = "cuda:0",
    global_seed: int = 42,
    base_std: float = 0.02,
    router_std: float = 0.001,
    hf_repo: str | None = None,
    upload: bool = False,
) -> dict[str, int | str]:
    assert_cluster_execution()
    config = load_config(config_path)
    model = config["model"]
    report = count_parameters(config)
    failures = validate_targets(config, report)
    if failures:
        raise RuntimeError(f"Parameter target validation failed: {', '.join(failures)}")

    manifest = build_manifest(model)
    max_shard_bytes = int(max_shard_gb * (1024**3))
    shard_groups = group_into_shards(manifest, max_shard_bytes)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    dtype = torch.bfloat16
    num_layers = int(model["num_layers"])
    weight_map: dict[str, str] = {}
    total_written = 0
    shard_count = len(shard_groups)

    for shard_idx, group in enumerate(shard_groups, start=1):
        shard_name = f"model-{shard_idx:05d}-of-{shard_count:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        for spec in group:
            tensors[spec.name] = init_tensor(
                spec.name,
                spec.shape,
                spec.kind,
                num_layers=num_layers,
                base_std=base_std,
                router_std=router_std,
                global_seed=global_seed,
                device=torch_device,
                dtype=dtype,
            )
            if torch.isnan(tensors[spec.name]).any() or torch.isinf(tensors[spec.name]).any():
                raise RuntimeError(f"Invalid values in tensor {spec.name}")
            weight_map[spec.name] = shard_name
            total_written += spec.nbytes

        cpu_tensors = {name: tensor.cpu() for name, tensor in tensors.items()}
        save_file(cpu_tensors, output_dir / shard_name)
        del tensors, cpu_tensors
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[knkf-init] wrote {shard_name} ({len(group)} tensors)")

    index = {
        "metadata": {"total_size": total_written},
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(build_hf_config(model, report.total_params, report.active_params), indent=2),
        encoding="utf-8",
    )
    (output_dir / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "max_new_tokens": 256}, indent=2),
        encoding="utf-8",
    )
    write_model_card(output_dir, str(config.get("name", "knk_vf_target")), report.total_params, report.active_params)

    init_meta = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "shards": shard_count,
        "tensors": len(manifest),
        "total_params": report.total_params,
        "active_params": report.active_params,
        "total_bytes": total_written,
        "dtype": "bfloat16",
        "device": device,
    }
    (output_dir / "init_metadata.json").write_text(json.dumps(init_meta, indent=2), encoding="utf-8")

    if upload:
        if not hf_repo:
            raise RuntimeError("--hf-repo is required with --upload")
        upload_to_hf(output_dir, hf_repo)
        init_meta["hf_repo"] = hf_repo

    return init_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-gb", type=float, default=4.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-std", type=float, default=0.02)
    parser.add_argument("--router-std", type=float, default=0.001)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    result = initialize_checkpoint(
        args.config,
        args.output,
        max_shard_gb=args.max_shard_gb,
        device=args.device,
        global_seed=args.seed,
        base_std=args.base_std,
        router_std=args.router_std,
        hf_repo=args.hf_repo,
        upload=args.upload,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
