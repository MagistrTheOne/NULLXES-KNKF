#!/usr/bin/env bash
# KUROTAMA lab: 500 Magistr-style dialogs -> KNK-VF Lab 38B LoRA on 1x H200.
# Uses LLaMA Factory as the runner only; no foreign pretrained checkpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export KNKF_CLUSTER_EXECUTION=1
export KNKF_ACCELERATOR=h200

MODEL_DIR="${MODEL_DIR:-/workspace/checkpoints/knk_vf_lab_38b}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/workspace/artifacts/knk_vf_tokenizer_lab/knk_vf.model}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "KNK checkpoint missing: ${MODEL_DIR}" >&2
  exit 1
fi

python scripts/build_kurotama_sft_corpus.py \
  --output-dir data/kurotama \
  --target-count 500 \
  --seed 42

python scripts/prepare_knk_llamafactory_checkpoint.py \
  --model-dir "${MODEL_DIR}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --repo-root "${REPO_ROOT}"

python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

model_dir = "/workspace/checkpoints/knk_vf_lab_38b"
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
assert config.model_type == "knk_vf"
assert tokenizer.encode("KUROTAMA", add_special_tokens=False)
print("knk_checkpoint_check: OK", config.model_type, tokenizer.__class__.__name__)
PY

llamafactory-cli train configs/train/llamafactory_kurotama_knk_h200.yaml

python - <<'PY'
import json
from pathlib import Path

path = Path("data/kurotama/kurotama_sharegpt.jsonl")
lines = path.read_text(encoding="utf-8").splitlines()
assert len(lines) == 500, f"expected 500 rows, got {len(lines)}"
sample = json.loads(lines[0])
text = " ".join(m["content"][:80] for m in sample["messages"])
assert "KUROTAMA" in text or "kurotama" in text.lower()
print("kurotama_corpus_check: OK (500 rows)")
print("DONE. Adapter: /workspace/outputs/kurotama-knk-vf-lab38b-lora")
PY
