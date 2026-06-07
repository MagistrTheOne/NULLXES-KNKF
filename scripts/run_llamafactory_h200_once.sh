#!/usr/bin/env bash
# One-shot NULLXES H200 run: bootstrap data -> LLaMA-Factory LoRA (~2h).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export KNKF_CLUSTER_EXECUTION=1
export KNKF_ACCELERATOR=h200

python scripts/build_nullxes_llamafactory_bundle.py \
  --repo-root "${REPO_ROOT}" \
  --output-dir data/llamafactory \
  --target-count 1800

llamafactory-cli train configs/train/llamafactory_nullxes_7b_h200.yaml

python - <<'PY'
import json
from pathlib import Path

sample = Path("data/llamafactory/nullxes_knk_vf_text.jsonl").read_text(encoding="utf-8").splitlines()[0]
text = json.loads(sample)["text"]
assert "<|system|>" in text and "<|user|>" in text and "<|assistant|>" in text
print("chat_style_check: OK")
print(text[:240].replace("\n", "\\n") + "...")
PY

echo "DONE. Adapter: /workspace/outputs/nullxes-llamafactory-7b-lora"
