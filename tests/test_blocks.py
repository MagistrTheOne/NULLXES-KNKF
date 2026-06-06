import torch

from knk.model.blocks import GQAAttention, RMSNorm, SwiGLU


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(16)
    x = torch.randn(2, 4, 16)
    assert norm(x).shape == x.shape


def test_gqa_attention_forward_shape():
    attn = GQAAttention(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        rope_theta=10000.0,
        local_window=8,
    )
    x = torch.randn(2, 8, 32)
    assert attn(x).shape == x.shape


def test_swiglu_forward_shape():
    ffn = SwiGLU(32, 64)
    x = torch.randn(2, 8, 32)
    assert ffn(x).shape == x.shape
