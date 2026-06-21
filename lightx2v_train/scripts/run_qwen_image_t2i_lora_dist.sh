#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7

torchrun \
--standalone \
--nproc_per_node=2 \
train.py --config configs/lora/qwen_image_lora_dist.yaml
