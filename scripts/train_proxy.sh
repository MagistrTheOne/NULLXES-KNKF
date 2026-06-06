#!/usr/bin/env bash
set -euo pipefail

python -m knk.train.proxy_trainer --config configs/train/proxy_h200_fsdp.yaml "$@"
