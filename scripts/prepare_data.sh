#!/usr/bin/env bash
set -euo pipefail

python -m knk.data.pipeline "$@"
