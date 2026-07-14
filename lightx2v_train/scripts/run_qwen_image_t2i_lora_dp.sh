#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

torchrun \
--standalone \
--nproc_per_node=2 \
train.py --config configs/train/flow/qwen_image_lora_dp.yaml
