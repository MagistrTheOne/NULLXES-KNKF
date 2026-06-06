---
pretty_name: "KNK-VF-153B Active-14B"
language:
  - en
  - ru
license: other
license_name: "NULLXES Proprietary / Research Use"
library_name: transformers
pipeline_tag: text-generation
tags:
  - nullxes
  - knkf
  - knk-vf
  - void-forged
  - kurotama-no-kami
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
  - h200
  - b300
  - megatron
model_creator: NULLXES
extra_gated_prompt: "This is a NULLXES initialization-only checkpoint. Not pretrained. Not for production inference."
model-index:
  - name: KNK-VF-153B-Init
    results: []
---

# KNK-VF-153B Active-14B

<p align="center">
  <strong>KUROTAMA-NO-KAMI VOID FORGED (KNK-VF)</strong><br/>
  NULLXES frontier Sparse Mixture-of-Experts initialization artifact
</p>

| | |
|---|---|
| **Owner** | NULLXES |
| **Contact** | [ceo@nullxes.com](mailto:ceo@nullxes.com) |
| **Program** | NULLXES KNKF |
| **Hub** | `MagistrTheOne/KNK-VF-153B` |
| **Codename** | KNK-VF-153B Active-14B |
| **Phase** | Init-only — prepared for aggressive datacenter pretrain |
| **Pretrained** | **No** |

## Summary

This repository publishes a **deterministically initialized**, **36-shard bf16 checkpoint** for the NULLXES KNKF frontier program. It is the **starting weight artifact** for large-scale pretraining on **NULLXES-owned H200/B300 datacenter clusters** — not a finished foundation model.

> **Warning:** Random-init inference produces meaningless text. Do not deploy for end users until pretrain + eval gates complete.

## Parameter report

| Metric | Value |
|--------|------:|
| Total parameters | 153,044,910,080 (~153.0B) |
| Active parameters / token | 14,129,561,600 (~14.1B) |
| Layers | 50 |
| Hidden size | 4,096 |
| Attention heads (GQA) | 32 / 8 KV |
| Routed experts | 128 (top-8 + 1 shared) |
| Context target | 262,144 tokens |
| Vocabulary target | 128,000 (EN / RU / code) |
| Precision | bfloat16 |
| Shards | 36 × ~8 GB safetensors |

## Architecture

- **Family:** NULLXES KNKF / KNK-VF VOID FORGED
- **Attention:** GQA + hybrid local/global pattern (window 4096, global every 4 layers)
- **FFN:** SwiGLU dense prefix (4 layers) + sigmoid-routed Sparse MoE (SME-BU load balance)
- **Position:** RoPE (`theta=1e6`) + YaRN scaling
- **Init policy:** Llama-NeoX-style residual (`base_std=0.02`, `router_std=0.001`, `global_seed=42`)
- **Config source:** `configs/model/knk_vf_153b_active14b.yaml`

## Purpose (NULLXES)

1. **Validate** MoE sharding, HF publishing, and cluster init pipeline on H200/B300.
2. **Bootstrap** aggressive target-scale pretraining on NULLXES datacenter infrastructure.
3. **Pair** with 128K EN/RU/code tokenizer + `knk_vf_chat_v2` bootstrap data before SFT.

## Execution policy

```text
KNKF_CLUSTER_EXECUTION=1
KNKF_ACCELERATOR=h200|b300
```

- Cluster-only execution — no local full-model materialization.
- Do not load all 153B parameters into single-node RAM/VRAM.
- Megatron-Core distributed pretrain is the intended consumer after proxy gates pass.

## Training roadmap

| Phase | Goal |
|-------|------|
| **0** ✅ | Sharded init checkpoint on Hub (this repo) |
| **1** | Train `knk_vf_tokenizer_128k` on EN/RU/code corpus |
| **2** | Bootstrap SFT pipeline validation (7B proxy, LLaMA-Factory) |
| **3** | Proxy MoE pretrain on H200 (routing + muP + data gates) |
| **4** | Aggressive 153B→1T pretrain on NULLXES datacenters (Megatron) |

## Bundled files

| File | Description |
|------|-------------|
| `model-00001..00036.safetensors` | Sharded bf16 weights |
| `model.safetensors.index.json` | Shard index |
| `config.json` | Architecture metadata |
| `init_metadata.json` | Init provenance |
| `tokenizer_config.json` | Tokenizer spec (no weights) |
| `special_tokens_map.json` | Special tokens |
| `chat_template.jinja` | `knk_vf_chat_v2` NULLXES identity template |

**Not included:** `tokenizer.model` (train separately).

## Limitations

- No pretraining or alignment — weights are randomly initialized.
- No production SLA, safety eval, or benchmark scores at this phase.
- Inference stacks (vLLM / Megatron) require NULLXES integration work.

## Citation

```bibtex
@misc{nullxes_knkf_153b_init_2026,
  title        = {KNK-VF-153B Active-14B: NULLXES VOID FORGED Initialization Checkpoint},
  author       = {NULLXES},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/MagistrTheOne/KNK-VF-153B}},
  note         = {Initialization-only. Contact: ceo@nullxes.com}
}
```

## Links

- **GitHub:** [NULLXES-KNKF](https://github.com/MagistrTheOne/NULLXES-KNKF)
- **Stable 1T target:** `configs/model/knk_vf_target_70b_active.yaml`
- **Ultra-frontier 3.5T:** `configs/model/knkf_kagutsuchi_tracks_3_5t.yaml`
- **Bootstrap data:** `configs/data/bootstrap_identity_tier0.yaml`

---
*NULLXES — KUROTAMA-NO-KAMI VOID FORGED. Prepared for aggressive frontier training on proprietary infrastructure.*
