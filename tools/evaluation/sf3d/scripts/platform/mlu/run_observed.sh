#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PLATFORM=cambricon_mlu
exec "${HOST_PYTHON_BIN:-python3}" "$EVAL_DIR/evaluate.py" campaign-mlu-observed --backend lightx2v "$@"
