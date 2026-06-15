#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=/data/nvme4/gushiqiao/new/diffusers/src:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=${CONFIG:-configs/infer/wan2_1_t2v_1_3b.yaml}

python3 infer.py --config "${CONFIG}"
