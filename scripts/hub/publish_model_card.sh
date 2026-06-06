#!/usr/bin/env bash
# Publish NULLXES KNK-VF Hub model card from repo source (after git pull).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${REPO_ROOT}/configs/hub/knk_vf_153b_hub.yaml"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required" >&2
  exit 1
fi

python - <<'PY' "${MANIFEST}" "${REPO_ROOT}"
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

manifest_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

card_src = repo_root / manifest["card"]["source"]
card_dst_name = manifest["card"]["destination"]
hub = manifest["hub"]

if not card_src.exists():
    raise SystemExit(f"Model card not found: {card_src}")

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=str(card_src),
    path_in_repo=card_dst_name,
    repo_id=hub["repo_id"],
    repo_type=hub["repo_type"],
    commit_message=hub.get("commit_message", "Update NULLXES model card"),
)

print(f"Published {card_src.name} -> {hub['repo_id']}/{card_dst_name}")
PY
