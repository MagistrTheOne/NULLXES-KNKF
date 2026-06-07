"""Transformers configuration for NULLXES KNK-VF checkpoints."""

from __future__ import annotations

from transformers import PretrainedConfig


class KNKVFConfig(PretrainedConfig):
    model_type = "knk_vf"

    def __init__(
        self,
        vocab_size: int = 128000,
        hidden_size: int = 3584,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 7,
        head_dim: int = 128,
        intermediate_size: int = 9626,
        expert_intermediate_size: int = 1408,
        num_routed_experts: int = 64,
        num_shared_experts: int = 1,
        routed_top_k: int = 4,
        dense_prefix_layers: int = 4,
        max_position_embeddings: int = 131072,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1.0e-6,
        tie_word_embeddings: bool = False,
        attention_pattern: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.expert_intermediate_size = expert_intermediate_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.routed_top_k = routed_top_k
        self.dense_prefix_layers = dense_prefix_layers
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attention_pattern = attention_pattern or {
            "type": "hybrid",
            "local_window": 4096,
            "global_every": 4,
        }

