#!/bin/bash
# full parameters train use fsdp2 by default
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
--standalone \
--nproc_per_node=4 \
train.py --config configs/full/flux2_klein_t2i.yaml
