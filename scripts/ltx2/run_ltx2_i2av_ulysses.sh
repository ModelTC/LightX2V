#!/bin/bash

# set path and first
lightx2v_path=/path/to/LightX2V
model_path=Lightricks/LTX-2

export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls ltx2 \
--task i2av \
--image_path "${lightx2v_path}/assets/inputs/imgs/woman.jpeg" \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_ulysses.json \
--prompt "A young woman with wavy, shoulder-length light brown hair is singing and dancing joyfully outdoors on a foggy day. She wears a cozy pink turtleneck sweater, swaying gracefully to the music with animated expressions and bright, piercing blue eyes. Her movements are fluid and energetic as she twirls and gestures expressively. A wooden fence and a misty, grassy field fade into the background, creating a dreamy atmosphere for her lively performance." \
--negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts." \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_ltx2_i2av_ulysses.mp4 \
--image_strength 1.0
