#!/usr/bin/env bash
set -euo pipefail

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

IFS=',' read -r -a VISIBLE_DEVICES <<< "${MUSA_VISIBLE_DEVICES}"

torchrun \
    --standalone \
    --nproc_per_node="${#VISIBLE_DEVICES[@]}" \
    train.py \
    --config "${CONFIG:-configs/train/flow/qwen_image_lora_dist.yaml}"
