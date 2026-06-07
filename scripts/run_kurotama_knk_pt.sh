#!/usr/bin/env bash
# KNK-only pretrain-like LoRA runner for LLaMA Factory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export KNKF_CLUSTER_EXECUTION=1
export KNKF_ACCELERATOR="${KNKF_ACCELERATOR:-h200}"

MODEL_DIR="${MODEL_DIR:-/workspace/checkpoints/knk_vf_lab_38b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/workspace/artifacts/knk_vf_tokenizer_lab/knk_vf.model}"
MODE="${1:-smoke}"

case "${MODE}" in
  smoke)
    TARGET_COUNT="${TARGET_COUNT:-1200}"
    CONFIG="configs/train/llamafactory_kurotama_knk_pt_smoke_h200.yaml"
    ;;
  week)
    TARGET_COUNT="${TARGET_COUNT:-20000}"
    CONFIG="configs/train/llamafactory_kurotama_knk_pt_week_h200.yaml"
    ;;
  *)
    echo "usage: $0 [smoke|week]" >&2
    exit 2
    ;;
esac

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "KNK checkpoint missing: ${MODEL_DIR}" >&2
  exit 1
fi

python scripts/build_kurotama_pretrain_corpus.py \
  --output-dir data/kurotama_pretrain \
  --target-count "${TARGET_COUNT}"

python scripts/prepare_knk_llamafactory_checkpoint.py \
  --model-dir "${MODEL_DIR}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --repo-root "${REPO_ROOT}"

rm -rf /root/.cache/huggingface/modules/transformers_modules/*knk_vf_lab_38b*

echo "Running ${MODE} PT with ${CONFIG}"
llamafactory-cli train "${CONFIG}"
