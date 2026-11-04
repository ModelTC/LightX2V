#!/bin/bash
set -euo pipefail
lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" "${lightx2v_path}/scripts/minimax_h3/inference/minimax_h3_t2av_mps_int8.py" "$@"
