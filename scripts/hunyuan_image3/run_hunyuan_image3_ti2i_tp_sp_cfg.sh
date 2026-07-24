#!/bin/bash
set -euo pipefail

# Uses the common TI2I launcher with TP=2, SP=2, and CFG=2.
export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export CONFIG_JSON="${CONFIG_JSON:-${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_ti2i_tp_sp_cfg.json}"
export SAVE_RESULT_PATH="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/hunyuan_image3_ti2i_tp2_sp2_cfg2.png}"

exec bash "${lightx2v_path}/scripts/hunyuan_image3/run_hunyuan_image3_ti2i_tp_sp.sh" "$@"
