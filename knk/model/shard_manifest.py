"""Flat tensor manifest for sharded KNK-VF checkpoint initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from knk.model.init_weights import InitKind


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    kind: InitKind
    nbytes: int


def _bytes(shape: tuple[int, ...], dtype_bytes: int = 2) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count * dtype_bytes


def _attention_specs(prefix: str, hidden: int, heads: int, kv_heads: int, head_dim: int) -> list[TensorSpec]:
    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    specs = [
        (f"{prefix}.self_attn.q_proj.weight", (q_out, hidden), InitKind.LINEAR_STD),
        (f"{prefix}.self_attn.k_proj.weight", (kv_out, hidden), InitKind.LINEAR_STD),
        (f"{prefix}.self_attn.v_proj.weight", (kv_out, hidden), InitKind.LINEAR_STD),
        (f"{prefix}.self_attn.o_proj.weight", (hidden, q_out), InitKind.LINEAR_RESIDUAL),
    ]
    return [TensorSpec(name, shape, kind, _bytes(shape)) for name, shape, kind in specs]


def _dense_ffn_specs(prefix: str, hidden: int, ffn_hidden: int) -> list[TensorSpec]:
    specs = [
        (f"{prefix}.mlp.gate_proj.weight", (ffn_hidden, hidden), InitKind.LINEAR_STD),
        (f"{prefix}.mlp.up_proj.weight", (ffn_hidden, hidden), InitKind.LINEAR_STD),
        (f"{prefix}.mlp.down_proj.weight", (hidden, ffn_hidden), InitKind.LINEAR_RESIDUAL),
    ]
    return [TensorSpec(name, shape, kind, _bytes(shape)) for name, shape, kind in specs]


def _moe_expert_specs(prefix: str, hidden: int, expert_ffn: int, expert_idx: int) -> list[TensorSpec]:
    base = f"{prefix}.mlp.experts.{expert_idx}"
    specs = [
        (f"{base}.gate_proj.weight", (expert_ffn, hidden), InitKind.LINEAR_STD),
        (f"{base}.up_proj.weight", (expert_ffn, hidden), InitKind.LINEAR_STD),
        (f"{base}.down_proj.weight", (hidden, expert_ffn), InitKind.LINEAR_RESIDUAL),
    ]
    return [TensorSpec(name, shape, kind, _bytes(shape)) for name, shape, kind in specs]


def build_manifest(model: dict[str, Any]) -> list[TensorSpec]:
    hidden = int(model["hidden_size"])
    layers = int(model["num_layers"])
    heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    head_dim = int(model["head_dim"])
    dense_prefix = int(model["dense_prefix_layers"])
    dense_ffn = int(model["ffn_hidden_size"])
    expert_ffn = int(model["expert_ffn_hidden_size"])
    experts = int(model["num_routed_experts"])
    shared = int(model.get("num_shared_experts", 1))
    vocab = int(model["vocab_size"])
    tied = bool(model.get("tie_embeddings", False))

    manifest: list[TensorSpec] = []
    manifest.append(
        TensorSpec("model.embed_tokens.weight", (vocab, hidden), InitKind.EMBEDDING, _bytes((vocab, hidden)))
    )

    for layer_idx in range(layers):
        prefix = f"model.layers.{layer_idx}"
        manifest.append(
            TensorSpec(
                f"{prefix}.input_layernorm.weight",
                (hidden,),
                InitKind.RMSNORM,
                _bytes((hidden,)),
            )
        )
        manifest.extend(_attention_specs(prefix, hidden, heads, kv_heads, head_dim))
        manifest.append(
            TensorSpec(
                f"{prefix}.post_attention_layernorm.weight",
                (hidden,),
                InitKind.RMSNORM,
                _bytes((hidden,)),
            )
        )

        if layer_idx < dense_prefix:
            manifest.extend(_dense_ffn_specs(prefix, hidden, dense_ffn))
            continue

        manifest.append(
            TensorSpec(
                f"{prefix}.mlp.router.gate.weight",
                (experts, hidden),
                InitKind.ROUTER,
                _bytes((experts, hidden)),
            )
        )
        for expert_idx in range(experts):
            manifest.extend(_moe_expert_specs(prefix, hidden, expert_ffn, expert_idx))
        for shared_idx in range(shared):
            manifest.extend(_moe_expert_specs(f"{prefix}.mlp.shared_experts", hidden, expert_ffn, shared_idx))

    manifest.append(TensorSpec("model.norm.weight", (hidden,), InitKind.RMSNORM, _bytes((hidden,))))
    if not tied:
        manifest.append(
            TensorSpec("lm_head.weight", (vocab, hidden), InitKind.EMBEDDING, _bytes((vocab, hidden)))
        )

    return manifest


def group_into_shards(manifest: list[TensorSpec], max_shard_bytes: int) -> list[list[TensorSpec]]:
    shards: list[list[TensorSpec]] = []
    current: list[TensorSpec] = []
    current_bytes = 0

    for spec in manifest:
        if current and current_bytes + spec.nbytes > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(spec)
        current_bytes += spec.nbytes

    if current:
        shards.append(current)
    return shards


def iter_manifest_stats(manifest: list[TensorSpec]) -> Iterator[tuple[str, int]]:
    total = sum(spec.nbytes for spec in manifest)
    yield ("tensors", len(manifest))
    yield ("total_bytes", total)
