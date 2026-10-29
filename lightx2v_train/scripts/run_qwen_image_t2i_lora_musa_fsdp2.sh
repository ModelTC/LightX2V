#!/usr/bin/env bash
set -euo pipefail

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --config configs/train/flow/qwen_image_lora_dist.yaml
