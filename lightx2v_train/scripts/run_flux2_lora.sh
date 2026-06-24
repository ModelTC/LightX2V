#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}

torchrun \
--standalone \
--nproc_per_node="${NPROC_PER_NODE}" \
train.py --config configs/train/flow/flux2_klein_lora.yaml
