#!/usr/bin/env bash
set -Eeuo pipefail

# Same command on all four ACP workers, eight GPUs per worker.
# One image count per global batch; randomly mixed orientations, no 16/16 quota.
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
# Ignore unrelated exported config/algorithm values from older experiments.
export H3_CONFIG_PATH="${H3_PDMD_COUNT_RANDOM_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_omni_imageonly_pdmd8_count_random_fsdp32.yaml}"
export H3_PDMD=true
unset H3_REF2AV_EXPECTED_ROWS
export H3_REF2AV_DMD_OUTPUT="${H3_REF2AV_DMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_ref2av_omni_imageonly_pdmd8_count_random_10k}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_ref2av_omni_imageonly_pdmd8_count_random_10k}"

# Reuse checked cache receipts, precision reporting, kernel setup and torchrun.
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_fsdp32_32gpu_acp.sh" "$@"
