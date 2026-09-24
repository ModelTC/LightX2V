#!/bin/bash

# Set paths.
lightx2v_path=path/to/LightX2V
model_path=path/to/BAGEL-7B-MoT

# Set environment variables.
export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls bagel \
  --task i2i \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/bagel/bagel_i2i.json" \
  --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
  --prompt "Change the scene to golden hour while preserving the main subject." \
  --seed 42 \
  --save_result_path "${lightx2v_path}/save_results/bagel_i2i.png"
