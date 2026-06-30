#!/bin/bash

# LongCat Image T2I 8-NPU inference script for Ascend NPU

lightx2v_path=${LIGHTX2V_PATH:-/data/wushuo1/LightX2V}
model_path=${MODEL_PATH:-/data/wushuo1/models/LongCat-Image}
npus=${NPUS:-8}
master_port=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/ascend_npu/dist/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --master_port="${master_port}" --nproc_per_node="${npus}" -m lightx2v.infer \
    --model_cls longcat_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/ascend_npu/dist/longcat_image_t2i.json" \
    --prompt "一只小猫躺在沙发上" \
    --negative_prompt "" \
    --save_result_path "${lightx2v_path}/save_results/longcat_image_t2i_dist8.png" \
    --seed 42
