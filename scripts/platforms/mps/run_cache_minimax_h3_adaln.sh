#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PLATFORM=mps
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export CONFIG_JSON="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/mps/minimax_h3_t2av_4step_512_22.json}"

exec bash "${lightx2v_path}/tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh" "$@"
