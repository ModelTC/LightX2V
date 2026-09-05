#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.1-I2V-14B-720P

export CUDA_VISIBLE_DEVICES=0
export DTYPE=FP16
export SENSITIVE_LAYER_DTYPE=FP16
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
  --model_cls wan2.1 \
  --task i2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/offload/disk/wan_i2v_phase_lazy_load_720p.json \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
  --seed 42 \
  --save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan_i2v_lazy_load.mp4
