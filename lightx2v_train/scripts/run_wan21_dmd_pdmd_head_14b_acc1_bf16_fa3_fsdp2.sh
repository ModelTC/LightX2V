#!/usr/bin/env bash
# Same acc1 comparison, FP32 master weights + BF16 compute, requiring FA3.
set -Eeuo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WAN_DMD_GRAD_ACCUM=1 WAN_DMD_PRECISION_PROFILE=bf16_fa3
exec bash "$script_dir/run_wan21_dmd_pdmd_head_14b_acc16_fsdp2.sh" "$@"
