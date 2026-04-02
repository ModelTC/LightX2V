#!/bin/bash

# set path and first
lightx2v_path=/data/nvme4/gushiqiao/LightX2V
model_path=/data/nvme0/gushiqiao/models/official_models/LTX-2/


export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls ltx2 \
--task i2av \
--image_path "/data/nvme4/gushiqiao/debug.png" \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_3.json \
--prompt "人物坐起身，双手撑床沿，双腿垂下床，身体前倾准备站起，镜头固定，表情从睡意转为清醒，动作流畅" \
--negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts." \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_ltx2_i2av3.mp4 \
--image_strength 1.0 \
--seed 0
