#!/bin/bash
# full parameters train use fsdp2 by default
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=${NPROC_PER_NODE:-4}

torchrun \
--standalone \
--nproc_per_node="${NPROC_PER_NODE}" \
train.py --config configs/train/flow/longcat_image_t2i.yaml
