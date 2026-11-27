#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${1:-}" == "--help" ]]; then
  echo "Run one frozen GenEval smoke input through PR #1754 on MLU590. This does not produce evaluation metrics."
  echo "Usage: PYTHON_BIN=/path/to/torch_mlu/python bash $0 --paths configs/mlu/paths.local.json --output /absolute/new-run-dir [--index 0]"
  exit 0
fi
exec "${PYTHON_BIN}" "$EVAL_DIR/scripts/platform/mlu/smoke.py" "$@"
