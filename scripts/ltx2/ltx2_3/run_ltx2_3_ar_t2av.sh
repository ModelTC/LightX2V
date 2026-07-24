#!/bin/bash

lightx2v_path=/data/nvme5/gushiqiao/codes/new/LightX2V
model_path=/data/nvme0/gushiqiao/models/official_models/LTX-2/

export CUDA_VISIBLE_DEVICES=6

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls ltx2_ar \
--task t2av \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_3_ar.json \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts." \
--save_result_path ${lightx2v_path}/save_results/output_ltx2_3_ar_t2av.mp4 \
--target_video_length 241
