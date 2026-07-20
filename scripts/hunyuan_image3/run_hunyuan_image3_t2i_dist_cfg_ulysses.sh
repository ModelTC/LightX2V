#!/bin/bash
set -euo pipefail

# Example (the CFG/SP sizes come from the selected JSON config):
#   bash scripts/dist_infer/run_hunyuan_image3_t2i_dist_cfg_ulysses.sh 0,1,2,3,4,5,6,7

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SP_ATTN_TYPE=ulysses
exec bash "${SCRIPT_DIR}/run_hunyuan_image3_t2i_dist_cfg_sp.sh" "$@"
