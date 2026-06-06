---
language:
- en
- ru
license: other
library_name: transformers
pipeline_tag: text-generation
tags:
- nullxes
- knkf
- knk-vf
- void-forged
- moe
- sparse-moe
- mixture-of-experts
- initialization
- random-init
- bf16
- long-context
- multilingual
- code
- enterprise
- text-generation
- not-for-all-audiences
model-index:
- name: KNK-VF-153B
  results: []
---

# KNK-VF-153B Active-14B

**KUROTAMA-NO-KAMI VOID FORGED (KNK-VF)** — NULLXES frontier Sparse Mixture-of-Experts foundation model artifact.

| Field | Value |
|-------|-------|
| **Owner** | NULLXES |
| **Contact** | [ceo@nullxes.com](mailto:ceo@nullxes.com) |
| **Hub repo** | `MagistrTheOne/KNK-VF-153B` |
| **Codename** | KNK-VF-153B Active-14B |
| **Status** | Initialization-only (not pretrained) |

## What this checkpoint is

This repository contains a **deterministically initialized**, **sharded bf16 weight artifact** for the NULLXES KNKF program. It is **not** a pretrained or instruction-tuned model. Outputs from raw inference are **not meaningful** for production use.

The checkpoint exists to:

1. Validate NULLXES MoE architecture, sharding, and Hugging Face publishing on H200/B300 infrastructure.
2. Serve as the **starting point for aggressive large-scale pretraining** on NULLXES-owned datacenter clusters.
3. Pair with the NULLXES 128K EN/RU/code tokenizer pipeline and bootstrap identity datasets before SFT.

## Architecture summary

- **Model family:** NULLXES KNKF / KNK-VF VOID FORGED
- **Total parameters:** ~153.0B
- **Active parameters per token:** ~14.1B
- **Layers:** 50
- **Hidden size:** 4096
- **Attention:** GQA (32 heads / 8 KV heads), hybrid local+global pattern
- **FFN:** SwiGLU dense prefix + sigmoid-routed Sparse MoE
- **Routed experts:** 128, top-8 per token + 1 shared expert
- **Context target:** 262,144 tokens (RoPE + YaRN)
- **Vocab target:** 128,000 (EN / RU / code)
- **Init policy:** Llama-NeoX-style residual scaling (`base_std=0.02`, `router_std=0.001`, `global_seed=42`)
- **Precision:** bfloat16
- **Sharding:** 36 safetensors shards, max ~8 GB per shard

## Execution policy

- **Cluster-only:** H200 or B300 accelerators under `KNKF_CLUSTER_EXECUTION=1`.
- **No local materialization** of the full 153B parameter tensor in RAM/VRAM.
- Intended for Megatron-Core / distributed pretrain after proxy validation gates pass.

## Training roadmap (NULLXES)

1. Train production tokenizer on EN/RU/code corpus (`knk_vf_tokenizer_128k`).
2. Proxy MoE pretrain on H200 cluster (architecture and routing validation).
3. **Aggressive target-scale pretraining** on NULLXES datacenter infrastructure.
4. SFT / alignment with `knk_vf_chat_v2` and enterprise safety bootstrap data.

## Limitations

- Random initialization only — **no pretraining data exposure**.
- No public inference guarantees until training and eval gates complete.
- Tokenizer weights (`tokenizer.model`) are not bundled; metadata and chat template are included.

## Citation / attribution

```
NULLXES KNKF — KNK-VF-153B Active-14B VOID FORGED initialization artifact.
Owner: NULLXES. Contact: ceo@nullxes.com
```

## Links

- GitHub: [NULLXES-KNKF](https://github.com/MagistrTheOne/NULLXES-KNKF)
- Stable 1T target config: `knk_vf_target_70b_active`
- Ultra-frontier target: `knkf_kagutsuchi_tracks_3_5t`
