"""Transformers-compatible KNK-VF causal LM.

This file is copied into initialized checkpoints as Hugging Face remote code.
It intentionally mirrors the tensor names emitted by knk.init.checkpoint:
model.layers.N.self_attn.*, model.layers.N.mlp.*, and lm_head.weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import GenerationConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_knk_vf import KNKVFConfig
except ImportError:  # local checkpoint remote-code execution
    from configuration_knk_vf import KNKVFConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos[..., : x.shape[-2], :]) + (rotate_half(x) * sin[..., : x.shape[-2], :])


class KNKVFAttention(nn.Module):
    def __init__(self, config: KNKVFConfig, layer_idx: int) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_group_size = config.num_attention_heads // config.num_key_value_heads
        self.local_window = None
        pattern = getattr(config, "attention_pattern", {}) or {}
        if pattern.get("type") == "hybrid" and (layer_idx + 1) % int(pattern.get("global_every", 4)) != 0:
            self.local_window = int(pattern.get("local_window", 4096))

        q_out = config.num_attention_heads * config.head_dim
        kv_out = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope(seq_len, hidden_states.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(seq_len, hidden_states.device))
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)

    def _mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(seq_len, device=device)[:, None]
        cols = torch.arange(seq_len, device=device)[None, :]
        keep = cols <= rows
        if self.local_window is not None:
            keep &= cols >= (rows - self.local_window + 1)
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        mask.masked_fill_(~keep, -math.inf)
        return mask


class KNKVFMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class RouterMetrics:
    router_entropy: torch.Tensor
    expert_utilization: torch.Tensor


class KNKVFRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, RouterMetrics]:
        logits = self.gate(x.float())
        scores = torch.sigmoid(logits)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        top_weights, top_indices = torch.topk(scores, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        flat_indices = top_indices.reshape(-1)
        tokens_per_expert = torch.bincount(flat_indices, minlength=self.num_experts).to(x.device)
        utilization = tokens_per_expert.float() / tokens_per_expert.sum().clamp_min(1)
        entropy = -(scores * scores.clamp_min(1.0e-9).log()).sum(dim=-1).mean()
        return top_indices, top_weights.to(x.dtype), RouterMetrics(entropy, utilization)


class KNKVFMoE(nn.Module):
    def __init__(self, config: KNKVFConfig) -> None:
        super().__init__()
        self.router = KNKVFRouter(config.hidden_size, config.num_routed_experts, config.routed_top_k)
        self.experts = nn.ModuleList(
            KNKVFMLP(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_routed_experts)
        )
        # Match the initialized checkpoint key path:
        # model.layers.N.mlp.shared_experts.mlp.experts.0.*
        self.shared_experts = KNKVFSharedExperts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        top_indices, top_weights, _ = self.router(flat_x)
        output = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts):
            token_idx, choice_idx = torch.where(top_indices == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat_x.index_select(0, token_idx))
            output.index_add_(0, token_idx, expert_out * top_weights[token_idx, choice_idx, None])

        for shared_expert in self.shared_experts.mlp.experts:
            output = output + shared_expert(flat_x)
        return output.reshape(original_shape)


class KNKVFSharedExpertMLP(nn.Module):
    def __init__(self, config: KNKVFConfig) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            KNKVFMLP(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_shared_experts)
        )


class KNKVFSharedExperts(nn.Module):
    def __init__(self, config: KNKVFConfig) -> None:
        super().__init__()
        self.mlp = KNKVFSharedExpertMLP(config)


class KNKVFDecoderLayer(nn.Module):
    def __init__(self, config: KNKVFConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = KNKVFAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if layer_idx < config.dense_prefix_layers:
            self.mlp = KNKVFMLP(config.hidden_size, config.intermediate_size)
        else:
            self.mlp = KNKVFMoE(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class KNKVFModel(nn.Module):
    def __init__(self, config: KNKVFConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(KNKVFDecoderLayer(config, idx) for idx in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class KNKVFPreTrainedModel(PreTrainedModel):
    config_class = KNKVFConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KNKVFDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        return None

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, KNKVFModel):
            module.gradient_checkpointing = value


class KNKVFForCausalLM(KNKVFPreTrainedModel):
    def __init__(self, config: KNKVFConfig) -> None:
        super().__init__(config)
        self.model = KNKVFModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.generation_config = GenerationConfig.from_model_config(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: object,
    ) -> CausalLMOutputWithPast:
        del attention_mask
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

