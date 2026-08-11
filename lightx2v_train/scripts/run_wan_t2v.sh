#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=1 \
    train.py \
    --config configs/train/flow/wan2_1_t2v_1_3b_lora.yaml
