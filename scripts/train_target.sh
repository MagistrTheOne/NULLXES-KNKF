#!/usr/bin/env bash
set -euo pipefail

python -m knk.train.megatron_launcher --config configs/train/target_megatron_h200.yaml "$@"
