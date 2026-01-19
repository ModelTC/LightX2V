#!/bin/bash

# set path and first
lightx2v_path=/data/nvme0/gushiqiao/models/code/LightX2V
model_path=/data/nvme0/gushiqiao/models/official_models/LTX-2
ltx2_repo=/data/nvme0/gushiqiao/models/code/LTX-2

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Add LTX-2 ltx_core to PYTHONPATH for VAE decoder
export PYTHONPATH=${ltx2_repo}/packages/ltx-core/src:$PYTHONPATH

python -m lightx2v.infer \
--model_cls ltx2 \
--task i2av \
--images "${lightx2v_path}/assets/inputs/imgs/img_0.jpg:0:1.0" \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/ltx2/ltx2.json \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts." \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_ltx2_i2av.mp4
