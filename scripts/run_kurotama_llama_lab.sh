#!/usr/bin/env bash
# Deprecated: foreign base-model runs are forbidden for KUROTAMA work.
set -euo pipefail

echo "This launcher is disabled: no foreign pretrained models in KUROTAMA work." >&2
echo "Use: bash scripts/run_kurotama_knk_lab.sh" >&2
exit 2
