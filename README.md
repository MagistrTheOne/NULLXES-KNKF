# NULLXES KNKF Frontier Models

NULLXES KNKF is a frontier Sparse Mixture-of-Experts foundation model program for
multilingual reasoning, long-context agentic workflows, and code generation.

Contact: `ceo@nullxes.com`

## Frontier Targets

- `KNK-VF`: the stable 1T-class VOID FORGED target, approximately 1.02T total
  parameters and 69.99B active parameters per token.
- `KNKF: Kurotama-no-Kami Kagutsuchi Tracks`: the ultra-frontier Cinder Tracks
  target, approximately 3.51T total parameters and 87.91B active parameters per
  token.

This repository starts with Phase 0 infrastructure:

- PyTorch reference model blocks for GQA, RoPE, SwiGLU, hybrid attention, and MoE routing.
- YAML-driven model, data, training, and inference configuration.
- A proxy training path for H200 validation before target-scale pretraining.
- Data preparation, tokenizer, evaluation, monitoring, inference, and runbook skeletons.

## Execution Policy

Do not launch local training, dry-runs, parameter validation, or evaluation jobs
for this project. KNKF is H200/B300-only operationally. Local work is limited to
editing source, configs, and documentation.

Target models are intentionally not launched directly. A smaller proxy must pass
muP calibration, data, routing, and evaluation gates on H200-class infrastructure
before Megatron-Core target pretraining is attempted.

## Project Layout

- `configs/` contains model, data, train, and inference YAMLs.
- `knk/model/` contains the PyTorch reference implementation and parameter counter.
- `knk/train/` contains proxy training and Megatron launch helpers.
- `knk/data/` contains corpus normalization, filtering, sharding, and sampling utilities.
- `knk/eval/` contains benchmark and long-context evaluation stubs.
- `docs/` captures architecture, data, training, and hardware decisions.
