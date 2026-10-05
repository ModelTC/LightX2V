#!/usr/bin/env bash
# Same three-way recipe, with no gradient accumulation, including the small head.
set -Eeuo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WAN_DMD_GRAD_ACCUM=1
exec bash "$script_dir/run_wan21_dmd_pdmd_head_14b_acc16_fsdp2.sh" "$@"
