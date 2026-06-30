#!/bin/bash

# LTX-2 S2V 8-NPU inference script for Ascend NPU

lightx2v_path=${LIGHTX2V_PATH:-/data/wushuo1/LightX2V}
model_path=${MODEL_PATH:-/data/wushuo1/models/LTX-2}
AUDIO_PATH=${AUDIO_PATH:-/data/wushuo1/ltx2_s2v_sample.wav}
npus=${NPUS:-8}
master_port=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/ascend_npu/dist/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --master_port="${master_port}" --nproc_per_node="${npus}" -m lightx2v.infer \
    --model_cls ltx2 \
    --task ltx2_s2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/ascend_npu/dist/ltx2_3.json" \
    --audio_path "${AUDIO_PATH}" \
    --prompt "A person speaks clearly in a quiet room, natural lighting, cinematic medium shot." \
    --negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts." \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_ltx2_s2v_dist8.mp4"
