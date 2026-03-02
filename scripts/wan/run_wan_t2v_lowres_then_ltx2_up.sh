#!/bin/bash

# set path firstly
lightx2v_path=/data/yongyang/Jin/LightX2V
model_path=/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B
ltx2_model_path=/data/yongyang/Jin/Lightricks/LTX-2
ltx2_vae_ckpt=/data/yongyang/Jin/Lightricks/LTX-2/ltx-2-19b-distilled.safetensors
ltx2_upsampler_ckpt=/data/yongyang/Jin/Lightricks/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

lowres_output=${lightx2v_path}/save_results/output_lightx2v_wan_t2v_lowres_ltx2.mp4
upscaled_output=${lightx2v_path}/save_results/output_lightx2v_wan_t2v_ltx2_upscaled.mp4

prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."

python -m lightx2v.infer \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan/wan_t2v_lowres_then_ltx2_up.json \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--save_result_path ${lowres_output}

python ${lightx2v_path}/tools/ltx2_upsample_video.py \
--input ${lowres_output} \
--output ${upscaled_output} \
--ltx2_vae_ckpt ${ltx2_vae_ckpt} \
--ltx2_upsampler_ckpt ${ltx2_upsampler_ckpt} \
--ltx2_transformer_ckpt ${ltx2_vae_ckpt} \
--ltx2_model_path ${ltx2_model_path} \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--device cuda \
--dtype bf16
