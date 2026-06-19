#!/bin/bash
# full parameters train use fsdp2 by default
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
--standalone \
--nproc_per_node=1 \
train.py --config configs/lora/longcat_image_lora.yaml
