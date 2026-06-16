#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

torchrun \
--standalone \
--nproc_per_node=2 \
train.py --config configs/lora/qwen_image_edit_2511_lora_dist.yaml
