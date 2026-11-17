#!/usr/bin/env bash
# Four-step variant of the shared Ref6k PDMD + GAN ACP launcher.
set -Eeuo pipefail
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_REF6K_CONFIG="${H3_REF6K_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_pdmd4_gan_from_cm1500_ref6k_fsdp32.yaml}"
export H3_REF6K_OUTPUT="${H3_REF6K_OUTPUT:-/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdmd4_gan_cm1500_ref6k_lr5e5_shift12_3_fake5_32gpu_20261117/train}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_ref2av_pdmd4_gan_cm1500_ref6k_20261117}"
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_ref6k_pdmd8_gan_fsdp32_32gpu_acp.sh" "$@"
