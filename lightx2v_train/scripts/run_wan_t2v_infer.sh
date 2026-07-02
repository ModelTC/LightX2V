#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=/data/nvme4/gushiqiao/new/diffusers/src:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CONFIG=${CONFIG:-configs/infer/wan2_2_ti2v_5b_tf_ar.yaml}

python3 infer.py --config "${CONFIG}"
