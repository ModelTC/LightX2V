#!/bin/bash
set -e

lightx2v_path=path/to/LightX2V
model_path=path/to/HunyuanVideo-1.5

export CUDA_VISIBLE_DEVICES=0,1,2,3
source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${lightx2v_path}/save_results/HY-WorldPlay

# Forward 15 latent steps, then backward 15 latent steps.
torchrun --nproc_per_node=4 -m lightx2v.infer \
    --model_cls worldplay_ar \
    --task i2v \
    --model_path ${model_path} \
    --config_json ${lightx2v_path}/configs/worldplay/worldplay_ar_i2v_480p_sp4.json \
    --prompt "A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water." \
    --image_path path/to/HY-WorldPlay/assets/img/test.png \
    --pose "w-15,s-15" \
    --num_frames 121 \
    --seed 42 \
    --save_result_path ${lightx2v_path}/save_results/HY-WorldPlay/worldplay_ar_sp4_test.mp4
