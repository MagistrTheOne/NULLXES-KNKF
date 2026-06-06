# KNKF Hardware Topology

KNKF planning is constrained to NVIDIA H200 or B300 class systems. No local
training, dry-runs, parameter validation, inference tests, or eval jobs are part
of the operating model.

## Proxy Topology

Proxy runs target a single 4x H200 node:

```text
strategy: FSDP
precision: bf16
gpus_per_node: 4
context curriculum: 8K -> 32K
```

This stage validates model blocks, routing, tokenizer compression, data quality,
loss behavior, and muP scaling before target training.

## Target Topology

The stable 1T-class KNK-VF starting Megatron-Core mapping is:

```text
TP = 4
PP = 12
EP = 32
CP = 2
DP = auto
```

Rationale:

- TP stays inside each 4-GPU H200 node.
- PP spreads 96 layers across pipeline stages.
- EP shards 256 routed experts across the cluster.
- CP is enabled for the 256K-token context path.

This is a planning default, not a final production launch contract. Before a long
run, profile memory and throughput on the actual cluster, then update this file
with measured microbatch size, activation checkpointing strategy, MFU, and
tokens-per-second.

## Ultra-Frontier Topology

`KNKF: Kurotama-no-Kami Kagutsuchi Tracks` targets roughly 3.51T total parameters
and a 1M-token context window.

Starting mapping for cluster profiling:

```text
TP = 8
PP = 16
EP = 64
CP = 4
DP = auto
```

Rationale:

- TP widens tensor shards for the 128-layer, 8192-hidden model.
- PP spreads 128 layers across pipeline stages.
- EP shards 512 routed experts across the cluster.
- CP is required for the 1M-token context target.

Prefer B300 for this target when available. H200 is allowed only with additional
profiling headroom, lower microbatch assumptions, and explicit acceptance of
lower MFU.

## Precision

- Proxy: bf16 matmul, fp32 loss and router gates
- Target warmup: bf16 matmul, fp32 loss and router gates
- Target steady state: FP8 only after the first stable 2,000 steps match the bf16
  proxy loss curve

## Required Monitoring

Alert on:

- loss spike above 2x running median
- NaN gradient norm
- sustained MFU drop above 20%
- expert utilization collapse
- skipped-token ratio above 5%
