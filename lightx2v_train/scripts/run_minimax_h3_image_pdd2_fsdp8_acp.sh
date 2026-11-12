#!/usr/bin/env bash
set -Eeuo pipefail

# The existing staged launcher keeps its PDMD default. This entry selects PDD.
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_IMAGE_TRAINING_METHOD=pdd
export H3_IMAGE_CONFIG="${H3_IMAGE_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/pdd/minimax_h3_image_pdd2_2048_fsdp8.yaml}"
export H3_PDD_OUTPUT="${H3_PDD_OUTPUT:-/mnt/devsft_afs_2/gushiqiao/outputs/h3_image_pdd2_10k_2048}"
export H3_PDD_MAX_ITERS="${H3_PDD_MAX_ITERS:-30000}"
# Route the shared launcher's log to the PDD output directory as well.
export H3_IMAGE_PDMD_OUTPUT="$H3_PDD_OUTPUT"
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_image_pdmd2_fsdp8_acp.sh" "$@"
