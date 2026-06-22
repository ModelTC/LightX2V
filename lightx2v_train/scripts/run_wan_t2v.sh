#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=/data/nvme4/gushiqiao/new/diffusers/src:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CONFIG=${CONFIG:-configs/full/wan2_1_t2v_1_3b_tf_chunkwise_full.yaml}

torchrun \
--standalone \
--nproc_per_node="${NPROC_PER_NODE}" \
train.py --config "${CONFIG}"
