#!/bin/bash

lightx2v_path=path/to/LightX2V
model_path=path/to/HunyuanVideo-1.5

export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${lightx2v_path}/save_results/HY-WorldPlay

# Right 31 latent steps.
python -m lightx2v.infer \
    --model_cls worldplay_ar \
    --task i2v \
    --model_path ${model_path} \
    --config_json ${lightx2v_path}/configs/worldplay/worldplay_ar_i2v_480p.json \
    --prompt "A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere." \
    --image_path path/to/HY-WorldPlay/assets/img/test.png \
    --pose "d-31" \
    --seed 1 \
    --save_result_path ${lightx2v_path}/save_results/HY-WorldPlay/worldplay_ar_test.mp4
