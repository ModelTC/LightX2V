#!/usr/bin/env bash
set -euo pipefail

export MACA_PATH="${MACA_PATH:-/opt/maca-3.7.1}"
export PATH="${MACA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --config configs/train/flow/qwen_image_lora_dist.yaml
