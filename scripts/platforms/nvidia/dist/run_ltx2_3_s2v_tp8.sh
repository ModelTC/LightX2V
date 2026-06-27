#!/bin/bash

# LTX-2 S2V 8-GPU Inference Script for NVIDIA
# System management interface: nvidia-smi

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme1/wushuo/LightX2V}
model_path=${MODEL_PATH:-/data/nvme1/wushuo/hf_models/models/LTX-2}
AUDIO_PATH=${AUDIO_PATH:-/data/nvme1/wushuo/ltx2_s2v_sample.wav}
gpus=${GPUS:-8}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/nvidia/dist/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --nproc_per_node="${gpus}" -m lightx2v.infer \
    --model_cls ltx2 \
    --task ltx2_s2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/nvidia/dist/ltx2_3_tp8.json" \
    --audio_path "${AUDIO_PATH}" \
    --prompt "A person speaks clearly in a quiet room, natural lighting, cinematic medium shot." \
    --negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts." \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_ltx2_s2v_dist8.mp4"
