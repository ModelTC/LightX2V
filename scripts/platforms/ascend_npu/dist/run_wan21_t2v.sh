#!/bin/bash

# Wan2.1 T2V 8-NPU inference script for Ascend NPU

lightx2v_path=${LIGHTX2V_PATH:-/data/wushuo1/LightX2V}
model_path=${MODEL_PATH:-/data/wushuo1/models/Wan2.1-T2V-1.3B}
npus=${NPUS:-8}
master_port=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/ascend_npu/dist/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --master_port="${master_port}" --nproc_per_node="${npus}" -m lightx2v.infer \
    --model_cls wan2.1 \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/ascend_npu/dist/wan_t2v.json" \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_wan_t2v_dist8.mp4"
