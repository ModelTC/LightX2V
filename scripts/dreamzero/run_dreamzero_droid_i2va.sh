#!/bin/bash

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme1/yongyang/nb/LightX2V}
model_path=${DREAMZERO_MODEL_PATH:-/data/nvme1/yongyang/nb/models/GEAR-Dreams/DreamZero-DROID}
wan_ckpt_dir=${WAN_CKPT_DIR:-/data/nvme0/models/Wan-AI/Wan2.1-I2V-14B-480P}
input_path=${DREAMZERO_INPUT_PATH:-/data/nvme1/yongyang/nb/dreamzero/debug_image}
save_path=${DREAMZERO_SAVE_PATH:-${lightx2v_path}/save_results/dreamzero_droid_i2va.mp4}
action_path=${DREAMZERO_ACTION_PATH:-${lightx2v_path}/save_results/dreamzero_droid_i2va.actions.npy}
prompt=${DREAMZERO_PROMPT:-Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls dreamzero \
--task i2va \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/dreamzero/dreamzero_droid_i2va.json \
--t5_original_ckpt ${wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth \
--t5_tokenizer_path ${wan_ckpt_dir}/google/umt5-xxl \
--clip_original_ckpt ${wan_ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--vae_path ${wan_ckpt_dir}/Wan2.1_VAE.pth \
--seed 1140 \
--prompt "${prompt}" \
--image_path ${input_path} \
--save_result_path ${save_path} \
--save_action_path ${action_path}
