#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$EVAL_DIR/evaluate.py" "$@"
