#!/usr/bin/env bash
# NULLXES one-shot: synthetic corpus -> train KNK tokenizer -> validate. No foreign FT.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export KNKF_CLUSTER_EXECUTION=1
export KNKF_ACCELERATOR=h200

python scripts/build_nullxes_llamafactory_bundle.py \
  --repo-root "${REPO_ROOT}" \
  --output-dir data/synthetic \
  --target-count 5000

python - <<'PY'
import json
from pathlib import Path

src = Path("data/synthetic/nullxes_knk_vf_text.jsonl")
out = Path("data/synthetic/pretrain_corpus.txt")
lines = []
for line in src.read_text(encoding="utf-8").splitlines():
    if line.strip():
        lines.append(json.loads(line)["text"].replace("\n", " "))
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"corpus_lines={len(lines)} -> {out}")
PY

python -m knk.tokenizer.train_tokenizer \
  --input data/synthetic/pretrain_corpus.txt \
  --output-prefix /workspace/artifacts/knk_vf_tokenizer_128k/knk_vf \
  --vocab-size 128000

python scripts/validate_tokenizer.py \
  --model /workspace/artifacts/knk_vf_tokenizer_128k/knk_vf.model

python -m knk.tokenizer.export_hf_tokenizer \
  --config configs/tokenizer/knk_vf_tokenizer_128k.yaml \
  --output /workspace/artifacts/knk_vf_tokenizer_hf

python scripts/count_params.py --config configs/model/knk_vf_lab_38b_active5b.yaml --validate

echo "TOKENIZER LAB DONE"
echo "tokenizer.model: /workspace/artifacts/knk_vf_tokenizer_128k/knk_vf.model"
echo "lab model config: configs/model/knk_vf_lab_38b_active5b.yaml (~38B total / ~5.3B active)"
