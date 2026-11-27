#!/usr/bin/env bash
set -euo pipefail
export PLATFORM=ascend_npu
export DTYPE=BF16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${PYTHON_BIN:-python}" "$HERE/evaluate.py" "$@"
