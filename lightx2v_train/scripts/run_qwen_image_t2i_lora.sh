#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
--standalone \
--nproc_per_node=1 \
train.py --config configs/train/flow/qwen_image_lora.yaml
