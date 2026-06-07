#!/usr/bin/env bash
# KUROTAMA lab: 500 Magistr-style dialogs -> Llama 3.1 8B LoRA on 1x H200.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export KNKF_CLUSTER_EXECUTION=1
export KNKF_ACCELERATOR=h200

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN required (Llama gated model + hub auth)" >&2
  exit 1
fi

python scripts/build_kurotama_sft_corpus.py \
  --output-dir data/kurotama \
  --target-count 500 \
  --seed 42

# LLaMA-Factory must be installed in the pod venv separately.
llamafactory-cli train configs/train/llamafactory_kurotama_llama_h200.yaml

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
print("sample_user:", json.loads(lines[0])["messages"][1]["content"][:120])
PY

echo "DONE. Adapter: /workspace/outputs/kurotama-llama31-8b-lora"
