#!/bin/bash

set -e

cd "$(dirname "$0")/.."

export PYTHONPATH=/data/nvme4/gushiqiao/new/diffusers/src:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CONFIG=${CONFIG:-configs/lora/wan2_1_t2v_1_3b_lora.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-/data/nvme5/gushiqiao/datatets/cache/wan2_1_t2v_1_3b_480x832x81}

torchrun \
--standalone \
--nproc_per_node="${NPROC_PER_NODE}" \
cache_wan_t2v.py \
--config "${CONFIG}" \
--split train \
--output_dir "${OUTPUT_DIR}"
