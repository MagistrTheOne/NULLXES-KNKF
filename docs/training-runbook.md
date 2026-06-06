# KNKF Training Runbook

## 1. Execution Boundary

Do not start any KNKF training, dry-run, parameter validation, benchmark, or
inference process on a local workstation. Operational execution is restricted to
approved H200/B300 cluster infrastructure.

Local development is limited to editing code, configs, and documentation.

## 2. Cluster Validation Gate

Validation must be run by cluster automation on H200/B300 nodes:

- unit tests
- parameter target validation
- proxy forward/backward validation
- Megatron memory profiling
- NCCL topology checks

Cluster jobs that execute KNKF code must set:

```text
KNKF_CLUSTER_EXECUTION=1
KNKF_ACCELERATOR=h200
```

or:

```text
KNKF_CLUSTER_EXECUTION=1
KNKF_ACCELERATOR=b300
```

Without those variables, training entrypoints must fail closed.

## 3. H200 Proxy Training

Proxy training starts from `configs/train/proxy_h200_fsdp.yaml` and must run on
4x H200 or larger approved infrastructure. The proxy validates model blocks,
tokenizer behavior, routing stability, data quality, loss behavior, and muP
scaling before any target-scale launch.

Monitor:

- loss
- gradient norm
- tokens per second
- router entropy
- expert utilization

Stop immediately on sustained loss spikes, NaNs, or expert collapse.

## 4. muP Calibration

Run width multipliers `{0.5, 1.0, 2.0}` on the proxy family and record loss and
learning rate. Use `knk.train.mup.transfer_learning_rate` to derive the first
target-scale learning rate candidate.

## 5. Target Training

The stable 1T-class target uses `configs/model/knk_vf_target_70b_active.yaml`.
The ultra-frontier Kagutsuchi Tracks target uses
`configs/model/knkf_kagutsuchi_tracks_3_5t.yaml`.

Before production training, cluster automation must run the Megatron memory
profiler and update `docs/hardware-topology.md` with the measured TP, PP, EP, CP,
DP, microbatch size, activation checkpointing strategy, MFU, and tokens/second.

## 6. Checkpoint Rules

Use distributed checkpoints containing:

- model weights
- optimizer state
- scheduler state
- data iterator state
- RNG state
- config JSON/YAML snapshot
- global step

Keep three rolling checkpoints and one evaluation milestone.
