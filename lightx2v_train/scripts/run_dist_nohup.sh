#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7

nohup \
torchrun \
--standalone \
--nproc_per_node=2 \
train.py --config configs/lora/qwen_image_lora_dist.yaml \
> qwen_image_lora_dist.log 2>&1 &


# You can kill all python processes by running:
# pkill -9 -f python
