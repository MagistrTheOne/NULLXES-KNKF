"""Parameter accounting for KNK-VF configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ParameterReport:
    attention_params: int
    dense_ffn_params: int
    moe_total_params: int
    moe_active_params: int
    embedding_params: int
    router_params: int
    total_params: int
    active_params: int

    def as_dict(self) -> dict[str, int]:
        return {
            "attention_params": self.attention_params,
            "dense_ffn_params": self.dense_ffn_params,
            "moe_total_params": self.moe_total_params,
            "moe_active_params": self.moe_active_params,
            "embedding_params": self.embedding_params,
            "router_params": self.router_params,
            "total_params": self.total_params,
            "active_params": self.active_params,
        }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def count_parameters(config: dict[str, Any]) -> ParameterReport:
    model = config["model"] if "model" in config else config
    vocab_size = int(model["vocab_size"])
    layers = int(model["num_layers"])
    hidden = int(model["hidden_size"])
    heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    head_dim = int(model["head_dim"])
    dense_prefix = int(model["dense_prefix_layers"])
    dense_ffn = int(model["ffn_hidden_size"])
    expert_ffn = int(model["expert_ffn_hidden_size"])
    experts = int(model["num_routed_experts"])
    shared = int(model.get("num_shared_experts", 1))
    top_k = int(model["routed_top_k"])
    tied = bool(model.get("tie_embeddings", False))

    q_params = hidden * heads * head_dim
    k_params = hidden * kv_heads * head_dim
    v_params = hidden * kv_heads * head_dim
    o_params = heads * head_dim * hidden
    attention_params = layers * (q_params + k_params + v_params + o_params)

    dense_ffn_params = dense_prefix * (3 * hidden * dense_ffn)
    moe_layers = layers - dense_prefix
    expert_params = 3 * hidden * expert_ffn
    moe_total_params = moe_layers * (experts + shared) * expert_params
    moe_active_params = moe_layers * (top_k + shared) * expert_params
    router_params = moe_layers * hidden * experts
    embedding_params = vocab_size * hidden * (1 if tied else 2)

    total_params = attention_params + dense_ffn_params + moe_total_params + router_params
    total_params += embedding_params
    active_params = attention_params + dense_ffn_params + moe_active_params + embedding_params
    active_params += router_params

    return ParameterReport(
        attention_params=attention_params,
        dense_ffn_params=dense_ffn_params,
        moe_total_params=moe_total_params,
        moe_active_params=moe_active_params,
        embedding_params=embedding_params,
        router_params=router_params,
        total_params=total_params,
        active_params=active_params,
    )


def validate_targets(config: dict[str, Any], report: ParameterReport) -> list[str]:
    targets = config.get("parameter_targets", {})
    failures: list[str] = []
    checks = {
        "active_params_min": report.active_params >= int(targets.get("active_params_min", 0)),
        "active_params_max": report.active_params <= int(targets.get("active_params_max", 10**30)),
        "total_params_min": report.total_params >= int(targets.get("total_params_min", 0)),
        "total_params_max": report.total_params <= int(targets.get("total_params_max", 10**30)),
    }
    for name, ok in checks.items():
        if not ok:
            failures.append(name)
    return failures
