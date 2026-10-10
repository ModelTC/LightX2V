#!/usr/bin/env bash
set -Eeuo pipefail

# Run identically on four ACP workers with eight GPUs each.
# Select the adapter-only FP32 variant without changing the original 32-GPU recipe.
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Ignore stale general/legacy config selectors; allow this recipe's own override.
export H3_LEGACY_SHIFT12_CONFIG="${H3_LEGACY_SHIFT12_LORA_FP32_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_lorafp32_fsdp32.yaml}"
export H3_REF2AV_DMD_OUTPUT="${H3_REF2AV_DMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_lorafp32_32gpu_100k}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_ref2av_omni_imageonly_dmd8_legacy_shift12_lorafp32_32gpu_100k}"

# Reuse cache-receipt verification, config reporting, model/kernel checks,
# --dry-run and the four-node torchrun path. This wrapper does not launch twice.
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp32_32gpu_acp.sh" "$@"
