#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

torchrun \
--standalone \
--nproc_per_node=1 \
train.py --config configs/lora/qwen_image_lora.yaml
