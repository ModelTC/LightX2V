#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

torchrun \
--standalone \
--nproc_per_node=1 \
train.py --config configs/lora/flux2_klein_lora.yaml
