#!/usr/bin/env bash
set -Eeuo pipefail

# Run once on one ACP node with eight GPUs. Each stage releases its models
# before the next stage starts; rerunning reuses completed preprocessing items.
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_IMAGE_MODEL_PATH="${H3_IMAGE_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-Image}"
export H3_IMAGE_TEXT_ENCODER_PATH="${H3_IMAGE_TEXT_ENCODER_PATH:-$H3_IMAGE_MODEL_PATH/text_encoder}"
export H3_IMAGE_PE_MODEL_PATH="${H3_IMAGE_PE_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-Image-PE}"
export H3_IMAGE_DATASET_DIR="${H3_IMAGE_DATASET_DIR:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_image_finet2i_10k_2048}"
export H3_IMAGE_PROMPTS="${H3_IMAGE_PROMPTS:-$H3_IMAGE_DATASET_DIR/prompts_raw.jsonl}"
export H3_IMAGE_PE_CACHE="${H3_IMAGE_PE_CACHE:-$H3_IMAGE_DATASET_DIR/pe}"
export H3_IMAGE_CACHE_DIR="${H3_IMAGE_CACHE_DIR:-$H3_IMAGE_DATASET_DIR/condition_cache}"
export H3_IMAGE_CACHE="$H3_IMAGE_CACHE_DIR/metadata.jsonl"
export H3_IMAGE_PDMD_OUTPUT="${H3_IMAGE_PDMD_OUTPUT:-/mnt/devsft_afs_2/gushiqiao/outputs/h3_image_pdmd2_10k_2048}"
export H3_IMAGE_PROMPT_COUNT="${H3_IMAGE_PROMPT_COUNT:-10000}"
export H3_IMAGE_MAX_ITERS="${H3_IMAGE_MAX_ITERS:-15000}"
export H3_IMAGE_PE_BACKEND="${H3_IMAGE_PE_BACKEND:-transformers}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ "${1:-}" != --dry-run ]]; then
    mkdir -p "$H3_IMAGE_PDMD_OUTPUT/logs"
    launch_log="$H3_IMAGE_PDMD_OUTPUT/logs/launch.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
    exec > >(tee -a "$launch_log") 2>&1
    echo "[h3-image] launcher log: $launch_log"
fi
cd "$H3_CODE_ROOT/lightx2v_train"
exec "$H3_PYTHON" -u "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_image_pdmd_stages.py" "$@"
