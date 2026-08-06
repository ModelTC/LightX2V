#!/usr/bin/env bash
set -euo pipefail

LIGHTX2V_ROOT="${LIGHTX2V_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${LIGHTX2V_ROOT}/scripts/server/sensenova/post_sensenova_vision.py" "$@"
