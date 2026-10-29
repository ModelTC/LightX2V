#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same command on four ACP workers (eight GPUs each).
# The source EMA is an initialization export, never a Turbo training resume.
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_STUDENT_INIT="${H3_STUDENT_INIT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1700_ema_init}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_FL2AV_CACHE="${H3_FL2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait}"
export H3_FL2AV_PDMD_OUTPUT="${H3_FL2AV_PDMD_OUTPUT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_pdmd8_from_turbo_dcm1700_ema_32gpu_20261029}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_fl2av_pdmd8_from_turbo_dcm1700_ema_32gpu_20261029}"
# A dedicated override avoids accidentally inheriting an older PDMD recipe.
export H3_FL2AV_PDMD_CONFIG="${H3_PDMD_FROM_TURBO_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_fl2av_3task_pdmd8_from_turbo_ema_shift6_3_r128_a128_adaln16_fsdp32.yaml}"

"$H3_PYTHON" "$H3_CODE_ROOT/lightx2v_train/scripts/minimax_h3_pdmd_from_turbo_preflight.py" "$H3_FL2AV_PDMD_CONFIG"
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_fl2av_pdmd_official_fsdp32_32gpu_acp.sh" 8 "$@"
