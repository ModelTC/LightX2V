#!/usr/bin/env bash
set -Eeuo pipefail

export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_IMAGE_TRAINING_METHOD=pdmd_gan
export H3_IMAGE_CONFIG="${H3_IMAGE_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_image_pdmd2_gan_2048_r128_a128_fsdp8.yaml}"
export H3_IMAGE_PDMD_OUTPUT="${H3_IMAGE_PDMD_OUTPUT:-/mnt/devsft_afs_2/gushiqiao/outputs/h3_image_pdmd2_gan_10k_2048}"
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_image_pdmd2_fsdp8_acp.sh" "$@"
