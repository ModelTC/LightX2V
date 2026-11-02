#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
diffusers_path=${DIFFUSERS_PATH:-/path/to/diffusers/src}
config=${CONFIG:-${lightx2v_path}/lightx2v_train/configs/train/dmd/wan2_1_t2v_1_3b_dmd_qat.yaml}

cd "${lightx2v_path}/lightx2v_train"
export PYTHONPATH="${diffusers_path}:${lightx2v_path}:${PYTHONPATH:-}"

nproc_per_node=${NPROC_PER_NODE:-8}

torchrun \
--standalone \
--nproc_per_node="${nproc_per_node}" \
train.py --config "${config}"
