#!/bin/bash
# full parameters train use fsdp2 by default
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
--standalone \
--nproc_per_node=4 \
train.py --config configs/full/qwen_image_t2i.yaml
