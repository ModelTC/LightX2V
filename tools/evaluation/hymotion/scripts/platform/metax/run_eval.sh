#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PLATFORM=metax_cuda
exec "${PYTHON_BIN:-python3}" "$EVAL_DIR/evaluate.py" campaign-metax --backend lightx2v --count 64 "$@"
