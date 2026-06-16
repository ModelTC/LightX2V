#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

torchrun \
--standalone \
--nproc_per_node=1 \
train.py --config configs/lora/qwen_image_edit_2511_lora.yaml
