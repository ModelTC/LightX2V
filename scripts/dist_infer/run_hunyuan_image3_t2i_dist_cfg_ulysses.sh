#!/bin/bash
set -euo pipefail

# Example (CFG=2, SP=2, PP=2):
#   SP_SIZE=2 bash scripts/dist_infer/run_hunyuan_image3_t2i_dist_cfg_ulysses.sh 0,1,2,3,4,5,6,7

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CFG_SIZE=2
export SP_ATTN_TYPE=ulysses
exec bash "${SCRIPT_DIR}/run_hunyuan_image3_t2i_dist_cfg_sp.sh" "$@"
