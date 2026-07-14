#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=/data/nvme4/gushiqiao/new/diffusers/src:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CONFIG=${CONFIG:-/data/nvme5/gushiqiao/codes/LightX2V/lightx2v_train/configs/train/dmd/wan2_1_t2v_1_3b_dmd.yaml}

torchrun \
--standalone \
--nproc_per_node="${NPROC_PER_NODE}" \
train.py --config "${CONFIG}"
