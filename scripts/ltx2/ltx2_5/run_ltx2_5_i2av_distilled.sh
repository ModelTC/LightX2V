#!/bin/bash
set -e

# set path firstly
lightx2v_path=/data/nvme0/gushiqiao/codes/LightX2V
model_path=/data/nvme0/gushiqiao/models/LTX-2.5
image_path=/data/nvme0/gushiqiao/codes/LightX2V/assets/inputs/imgs/girl.png
prompt="A cheerful stylized little girl in a red traditional Chinese dress smiles and gently waves as the camera slowly dollies out, clean white background"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls ltx2_5 \
--task i2av \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/ltx2/ltx2_5_distilled_8gpu.json" \
--image_path "${image_path}" \
--image_frame_idx 0 \
--image_strength 1.0 \
--prompt "${prompt}" \
--num_frames 121 \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/ltx2_5_distilled_i2av.mp4"
