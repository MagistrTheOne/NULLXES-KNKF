"""Reference KNK-VF causal language model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from knk.model.blocks import GQAAttention, RMSNorm, SwiGLU
from knk.model.hybrid_attention import HybridAttentionPattern
from knk.model.moe import RouterMetrics, SparseMoE


@dataclass
class KNKVFConfig:
    vocab_size: int = 128000
    max_position_embeddings: int = 32768
    num_layers: int = 24
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    dense_prefix_layers: int = 2
    ffn_hidden_size: int = 5504
    expert_ffn_hidden_size: int = 8192
    num_routed_experts: int = 64
    num_shared_experts: int = 1
    routed_top_k: int = 4
    norm_eps: float = 1.0e-6
    tie_embeddings: bool = False
    attention_pattern: dict[str, int | str] = field(
        default_factory=lambda: {"type": "hybrid", "local_window": 4096, "global_every": 4}
    )

    @classmethod
    def from_model_dict(cls, data: dict) -> "KNKVFConfig":
        accepted = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in data.items() if key in accepted}
        return cls(**filtered)


class DecoderLayer(nn.Module):
    def __init__(self, config: KNKVFConfig, layer_idx: int) -> None:
        super().__init__()
        pattern = HybridAttentionPattern(
            local_window=int(config.attention_pattern.get("local_window", 4096)),
            global_every=int(config.attention_pattern.get("global_every", 4)),
        )
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attention = GQAAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            local_window=pattern.local_window_for_layer(layer_idx),
        )
        self.post_attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if layer_idx < config.dense_prefix_layers:
            self.feed_forward: nn.Module = SwiGLU(config.hidden_size, config.ffn_hidden_size)
            self.is_moe = False
        else:
            self.feed_forward = SparseMoE(
                hidden_size=config.hidden_size,
                expert_ffn_hidden_size=config.expert_ffn_hidden_size,
                num_routed_experts=config.num_routed_experts,
                top_k=config.routed_top_k,
                num_shared_experts=config.num_shared_experts,
            )
            self.is_moe = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouterMetrics | None]:
        x = x + self.attention(self.input_norm(x))
        ff_input = self.post_attention_norm(x)
        if self.is_moe:
            ff_out, metrics = self.feed_forward(ff_input)
        else:
            ff_out = self.feed_forward(ff_input)
            metrics = None
        return x + ff_out, metrics


class KNKVFForCausalLM(nn.Module):
    def __init__(self, config: KNKVFConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config, idx) for idx in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[RouterMetrics]]:
        hidden_states = self.embed_tokens(input_ids)
        router_metrics: list[RouterMetrics] = []
        for layer in self.layers:
            hidden_states, metrics = layer(hidden_states)
            if metrics is not None:
                router_metrics.append(metrics)
        logits = self.lm_head(self.norm(hidden_states))
        output: dict[str, torch.Tensor | list[RouterMetrics]] = {
            "logits": logits,
            "router_metrics": router_metrics,
        }
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
            output["loss"] = loss
        return output
