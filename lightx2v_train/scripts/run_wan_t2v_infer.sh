#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
diffusers_path=${DIFFUSERS_PATH:-/path/to/diffusers/src}
config=${CONFIG:-${lightx2v_path}/lightx2v_train/configs/infer/wan2_2_ti2v_5b_tf_ar.yaml}

cd "${lightx2v_path}/lightx2v_train"
export PYTHONPATH="${diffusers_path}:${lightx2v_path}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

python3 infer.py --config "${config}"
