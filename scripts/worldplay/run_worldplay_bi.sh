#!/bin/bash
# Run WorldPlay BI model inference with LightX2V

export PYTHONPATH="${PYTHONPATH}:/workspace/LightX2V"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Model paths
MODEL_PATH=/data/nvme1/models/hunyuan/hf_cache/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
BI_ACTION_MODEL_PATH=/data/nvme1/models/hunyuan/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors

# Input parameters
PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere.'
IMAGE_PATH=/workspace/HY-WorldPlay/assets/img/test.png
POSE='s-31'  # Camera trajectory: backward movement for 31 latents
SEED=1

# Output
OUTPUT_PATH=/workspace/LightX2V/save_results/HY-WorldPlay/worldplay_bi_test.mp4

# Create output directory
mkdir -p $(dirname $OUTPUT_PATH)

# Run inference
python /workspace/LightX2V/lightx2v/infer.py \
    --model_cls worldplay_bi \
    --task i2v \
    --model_path $MODEL_PATH \
    --config_json /workspace/LightX2V/configs/worldplay/worldplay_bi_i2v_480p.json \
    --prompt "$PROMPT" \
    --image_path $IMAGE_PATH \
    --pose "$POSE" \
    --action_ckpt $BI_ACTION_MODEL_PATH \
    --seed $SEED \
    --save_result_path $OUTPUT_PATH

echo "Video saved to: $OUTPUT_PATH"
