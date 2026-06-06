"""Sparse Mixture-of-Experts layers for KNK-VF."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from knk.model.blocks import SwiGLU


@dataclass
class RouterMetrics:
    tokens_per_expert: torch.Tensor
    expert_utilization: torch.Tensor
    router_entropy: torch.Tensor
    skipped_tokens: torch.Tensor


class NormalizedSigmoidRouter(nn.Module):
    """Independent sigmoid routing normalized across experts, then top-k selected."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
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
        skipped = torch.zeros((), device=x.device)
        metrics = RouterMetrics(tokens_per_expert, utilization, entropy, skipped)
        return top_indices, top_weights.to(x.dtype), metrics


class SparseMoE(nn.Module):
    """Reference token-dispatch MoE with routed experts plus always-on shared expert."""

    def __init__(
        self,
        hidden_size: int,
        expert_ffn_hidden_size: int,
        num_routed_experts: int,
        top_k: int,
        num_shared_experts: int = 1,
    ) -> None:
        super().__init__()
        self.router = NormalizedSigmoidRouter(hidden_size, num_routed_experts, top_k)
        self.experts = nn.ModuleList(
            SwiGLU(hidden_size, expert_ffn_hidden_size) for _ in range(num_routed_experts)
        )
        self.shared_experts = nn.ModuleList(
            SwiGLU(hidden_size, expert_ffn_hidden_size) for _ in range(num_shared_experts)
        )
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouterMetrics]:
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        top_indices, top_weights, metrics = self.router(flat_x)
        output = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts):
            token_idx, choice_idx = torch.where(top_indices == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat_x.index_select(0, token_idx))
            output.index_add_(0, token_idx, expert_out * top_weights[token_idx, choice_idx, None])

        for shared_expert in self.shared_experts:
            output = output + shared_expert(flat_x)

        return output.reshape(original_shape), metrics


def router_z_loss(logits: torch.Tensor, coefficient: float) -> torch.Tensor:
    if coefficient <= 0:
        return logits.new_zeros(())
    return coefficient * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


def load_balance_aux_loss(tokens_per_expert: torch.Tensor, coefficient: float = 1.0e-2) -> torch.Tensor:
    if coefficient <= 0:
        return tokens_per_expert.new_zeros((), dtype=torch.float32)
    probs = tokens_per_expert.float() / tokens_per_expert.sum().clamp_min(1)
    uniform = torch.full_like(probs, 1.0 / probs.numel())
    return coefficient * F.mse_loss(probs, uniform)
