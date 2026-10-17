#!/usr/bin/env bash
set -Eeuo pipefail

# Matched video/audio rollout and inference shifts 6/3, with the source PDMD loop.
steps="${1:?Pass the rollout step count (4 or 8), optionally followed by --dry-run}"
if [[ ! "$steps" =~ ^(4|8)$ ]]; then
    echo "Usage: $0 {4|8} [--dry-run]" >&2
    exit 1
fi
shift
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_FL2AV_PDMD_CONFIG="$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_fl2av_pdmd${steps}_shift6_3_fsdp32.yaml"
export H3_FL2AV_PDMD_OUTPUT="${H3_FL2AV_PDMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_fl2av_3task_pdmd${steps}_shift6_3_lorafp32_32gpu}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_fl2av_3task_pdmd${steps}_shift6_3_lorafp32_32gpu}"
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_fl2av_pdmd_official_fsdp32_32gpu_acp.sh" "$steps" "$@"
