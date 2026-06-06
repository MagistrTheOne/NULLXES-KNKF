# NULLXES KNKF Architecture

KNKF is the NULLXES frontier Sparse Mixture-of-Experts foundation model family for
enterprise multilingual reasoning, code generation, and long-context workflows.

## Stable Target Model

The target configuration lives in `configs/model/knk_vf_target_70b_active.yaml`.
The current validated parameter report is:

- Active parameters per token: `69,992,265,728`
- Total parameters: `1,019,577,202,688`
- Routed experts: `256`
- Routed experts per token: `8`
- Shared experts per MoE layer: `1`
- Context target: `262,144` tokens

## Ultra-Frontier Target

The aggressive frontier configuration lives in
`configs/model/knkf_kagutsuchi_tracks_3_5t.yaml`.

`KNKF: Kurotama-no-Kami Kagutsuchi Tracks` uses Kagutsuchi, the Japanese fire
deity, as the mythological reference for the `Cinder Tracks` codename.

Estimated parameter report:

- Active parameters per token: `87,912,611,840`
- Total parameters: `3,512,477,941,760`
- Layers: `128`
- Hidden size: `8192`
- Routed experts: `512`
- Routed experts per token: `8`
- Shared experts per MoE layer: `1`
- Context target: `1,048,576` tokens

This is intentionally beyond the 1T-class KNK-VF baseline. It is a cluster-only
research target and must not be launched locally.

## Core Blocks

Each decoder layer follows the pre-norm pattern:

```text
x -> RMSNorm -> GQA + RoPE -> residual
  -> RMSNorm -> Dense SwiGLU or Sparse MoE -> residual
```

The stable target uses six dense prefix layers. Kagutsuchi Tracks uses eight dense
prefix layers to stabilize optimization before routed expert dispatch begins.
Later layers use normalized sigmoid routing with top-k expert selection and one
always-on shared expert.

## Attention

The attention stack uses grouped-query attention with an 8:1 query-to-KV ratio.
For the stable target:

- `88` query heads
- `11` KV heads
- `128` head dimension
- Hybrid local/global pattern with one full-attention layer every four layers

This keeps KV-cache growth manageable for 256K-token workflows while preserving
periodic full-context integration.

For Kagutsuchi Tracks:

- `64` query heads
- `8` KV heads
- `128` head dimension
- 1M token context target with hybrid local/global attention

## Routing

The reference router computes independent sigmoid expert scores, normalizes them,
then dispatches each token to the top eight routed experts. Router metrics include
tokens per expert, utilization, entropy, and skipped-token placeholders so the
training loop can detect expert collapse early.

## Scale Path

KNK-VF must not be trained at target scale cold. The repository includes a 3B
active proxy configuration for H200 FSDP validation, followed by muP calibration
and only then Megatron-Core target training.
