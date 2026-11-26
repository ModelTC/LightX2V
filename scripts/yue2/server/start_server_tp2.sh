#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/models/YuE2-3B
config_json=${1:-${lightx2v_path}/configs/yue2/yue2_tp2.json}
host=127.0.0.1
port=${2:-8000}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=4
source "${lightx2v_path}/scripts/base/base.sh"

exec torchrun --standalone --nproc_per_node=2 -m lightx2v.server \
    --model_cls yue2 \
    --task t2a \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --host "$host" \
    --port "$port"
