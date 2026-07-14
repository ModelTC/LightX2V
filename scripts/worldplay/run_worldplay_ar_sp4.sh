#!/bin/bash
# Run WorldPlay AR model inference with 4-GPU sequence parallel
set -e

export PYTHONPATH="${PYTHONPATH}:/workspace/LightX2V"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Model paths
MODEL_PATH=/data/nvme1/models/hunyuan/HunyuanVideo-1.5
AR_ACTION_MODEL_PATH=/data/nvme1/models/hunyuan/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors

# Input parameters
PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water.'
IMAGE_PATH=/workspace/HY-WorldPlay/assets/img/test.png
POSE='w-15,s-15'  # Forward 15 frames, then backward 15 frames. Auto target_video_length=121
SEED=42

# Output
OUTPUT_PATH=/workspace/LightX2V/save_results/HY-WorldPlay/worldplay_ar_sp4_test.mp4

mkdir -p $(dirname $OUTPUT_PATH)

torchrun --nproc_per_node=4 -m lightx2v.infer \
    --model_cls worldplay_ar \
    --task i2v \
    --model_path $MODEL_PATH \
    --config_json /workspace/LightX2V/configs/worldplay/worldplay_ar_i2v_480p_sp4.json \
    --prompt "$PROMPT" \
    --image_path $IMAGE_PATH \
    --pose "$POSE" \
    --action_ckpt $AR_ACTION_MODEL_PATH \
    --seed $SEED \
    --save_result_path $OUTPUT_PATH

echo "Video saved to: $OUTPUT_PATH"
