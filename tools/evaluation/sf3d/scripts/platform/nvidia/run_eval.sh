#!/usr/bin/env bash
set -euo pipefail
EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$EVAL_DIR/run_eval.sh" "$@"
