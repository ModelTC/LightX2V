#!/bin/bash
# full parameters train use fsdp2 by default
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=${NPROC_PER_NODE:-4}

torchrun \
--standalone \
--nproc_per_node="${NPROC_PER_NODE}" \
train.py --config configs/train/flow/flux2_klein_t2i.yaml
