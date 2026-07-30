#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}

torchrun \
--standalone \
--nproc_per_node="${NPROC_PER_NODE}" \
train.py --config configs/train/fastwam/libero_plus_uncond_2cam224_8gpu.yaml
