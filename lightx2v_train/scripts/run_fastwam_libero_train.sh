#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}

torchrun \
--standalone \
--nproc_per_node="${NPROC_PER_NODE}" \
train.py --config configs/train/fastwam/libero_uncond_2cam224.yaml
