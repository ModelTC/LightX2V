#!/bin/bash

set -e

lightx2v_path=${LIGHTX2V_PATH:-/mnt/devsft_afs_1/gushiqiao/LightX2V}
model_path=${MODEL_PATH:-/models/seko_ar}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/seko_talk/ar/seko_talk_ar_kv_dist.json}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
nproc_per_node=${NPROC_PER_NODE:-8}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node="${nproc_per_node}" -m lightx2v.server \
    --model_cls seko_talk_ar \
    --task rs2v \
    --model_path "${model_path}" \
    --config_json "${config_json}" \
    --host "${host}" \
    --port "${port}"

echo "Service stopped"
