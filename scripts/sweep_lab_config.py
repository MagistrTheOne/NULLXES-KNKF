"""Sweep KNK-VF lab configs around 30-40B total."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knk.model.param_counter import count_parameters

base = {
    "vocab_size": 128000,
    "max_position_embeddings": 131072,
    "rope_theta": 1000000.0,
    "rope_scaling": "yarn",
    "attention_pattern": {"type": "hybrid", "local_window": 4096, "global_every": 4},
    "dense_prefix_layers": 4,
    "router": {"type": "normalized_sigmoid", "gate_dtype": "fp32", "load_balance": "smebu"},
    "norm": "rmsnorm",
    "norm_eps": 1e-6,
    "use_bias": False,
    "tie_embeddings": False,
    "z_loss_coefficient": 1e-5,
}
candidates = []
for layers in [32, 36, 40]:
    for hidden in [2816, 3072, 3200, 3584]:
        heads = hidden // 128
        kv = max(heads // 4, 2)
        for experts in [40, 48, 56, 64]:
            for expert_ffn in [1408, 1536, 1664, 1792]:
                for top_k in [4, 6]:
                    dense_ffn = int(hidden * 2.6875)
                    m = dict(base)
                    m.update(
                        {
                            "num_layers": layers,
                            "hidden_size": hidden,
                            "num_attention_heads": heads,
                            "num_key_value_heads": kv,
                            "head_dim": 128,
                            "ffn_hidden_size": dense_ffn,
                            "expert_ffn_hidden_size": expert_ffn,
                            "num_routed_experts": experts,
                            "num_shared_experts": 1,
                            "routed_top_k": top_k,
                        }
                    )
                    r = count_parameters({"model": m})
                    if 30e9 <= r.total_params <= 45e9 and 4e9 <= r.active_params <= 9e9:
                        candidates.append((r.total_params, r.active_params, m))

candidates.sort(key=lambda x: abs(x[0] - 38e9))
for total, active, m in candidates[:10]:
    print(
        f"total={total/1e9:.2f}B active={active/1e9:.2f}B "
        f"layers={m['num_layers']} h={m['hidden_size']} "
        f"exp={m['num_routed_experts']} eff={m['expert_ffn_hidden_size']} top={m['routed_top_k']}"
    )
