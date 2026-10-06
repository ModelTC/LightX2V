#!/usr/bin/env bash
# Fresh LoRA-student/full-fake comparison; preserve the acc1 BF16/FA3 recipe.
set -Eeuo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WAN_DMD_STUDENT_TRAIN_TYPE=lora
export WAN_DMD_GRAD_ACCUM=1 WAN_DMD_PRECISION_PROFILE=bf16_fa3
exec bash "$script_dir/run_wan21_dmd_pdmd_head_14b_acc16_fsdp2.sh" "$@"
