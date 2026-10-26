#!/bin/bash

# Set paths.
lightx2v_path=/path/to/LightX2V
model_path=/path/to/Matrix-Game-3.0

export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls wan2.2_matrix_game3 \
    --task i2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/matrix_game3/matrix_game3_base.json \
    --prompt "a city street scene with cars and pedestrians" \
    --negative_prompt "Vibrant colors, overexposure, static, blurred details, subtitles, style, artwork, painting, still image, overall grayness, worst quality, low quality, JPEG compression residue, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background, three legs, crowded background, walking backwards" \
    --image_path /path/to/image.png \
    --action_path /path/to/action \
    --save_result_path ${lightx2v_path}/save_results/matrix_game3_base.mp4 \
    --seed 42
