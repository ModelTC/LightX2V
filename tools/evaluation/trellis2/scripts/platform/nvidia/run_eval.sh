#!/usr/bin/env bash
set -euo pipefail
project_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export EVAL_PLATFORM=nvidia
exec python3 "${project_path}/evaluate.py" --platform nvidia "$@"
