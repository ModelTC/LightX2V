#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    train.py \
    --config configs/train/flow/qwen_image_lora_ascend_fsdp2_cache.yaml
