"""Core decoder-only transformer blocks for KNK-VF."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by modern decoder LLMs."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


@dataclass(frozen=True)
class RopeConfig:
    head_dim: int
    max_position_embeddings: int
    theta: float = 1_000_000.0


class RotaryEmbedding(nn.Module):
    """RoPE cache with long-context theta support."""

    def __init__(self, config: RopeConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = config.max_position_embeddings

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


class SwiGLU(nn.Module):
    """Bias-free SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GQAAttention(nn.Module):
    """Grouped-query causal attention with optional sliding-window masking."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        local_window: int | None = None,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if hidden_size != num_attention_heads * head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.local_window = local_window
        self.kv_group_size = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.rope = RotaryEmbedding(
            RopeConfig(
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                theta=rope_theta,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_key_value_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope(seq_len, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)
        mask = self._attention_mask(seq_len, x.device)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)

    def _attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(seq_len, device=device)[:, None]
        cols = torch.arange(seq_len, device=device)[None, :]
        causal = cols <= rows
        if self.local_window is not None:
            causal &= cols >= (rows - self.local_window + 1)
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        mask.masked_fill_(~causal, -math.inf)
        return mask
