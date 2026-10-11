#!/bin/bash

# Set paths.
lightx2v_path=path/to/LightX2V
model_path=path/to/Cosmos3-Nano

# Set environment variables.
export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls cosmos3 \
  --task i2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/cosmos3/cosmos3_nano_omni.json" \
  --prompt "${model_path}/assets/example_i2v_prompt.json" \
  --negative_prompt "${model_path}/assets/negative_prompt.json" \
  --image_path "${model_path}/assets/example_i2v_input.jpg" \
  --save_result_path "${lightx2v_path}/save_results/cosmos3_nano_omni_i2v.mp4" \
  --seed 17
