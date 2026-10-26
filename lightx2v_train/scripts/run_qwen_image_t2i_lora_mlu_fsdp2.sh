#!/usr/bin/env bash
set -euo pipefail

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_MLU_ALLOC_CONF="${PYTORCH_MLU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --config configs/train/flow/qwen_image_lora_dist.yaml
