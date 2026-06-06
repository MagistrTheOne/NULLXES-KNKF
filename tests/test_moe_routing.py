import torch

from knk.model.moe import SparseMoE


def test_sparse_moe_returns_metrics():
    moe = SparseMoE(
        hidden_size=32,
        expert_ffn_hidden_size=64,
        num_routed_experts=4,
        top_k=2,
        num_shared_experts=1,
    )
    x = torch.randn(2, 5, 32)
    y, metrics = moe(x)
    assert y.shape == x.shape
    assert metrics.tokens_per_expert.numel() == 4
    assert metrics.expert_utilization.numel() == 4
    assert metrics.router_entropy.ndim == 0
