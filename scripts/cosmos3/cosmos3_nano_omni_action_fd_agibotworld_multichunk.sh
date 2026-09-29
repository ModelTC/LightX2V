#!/bin/bash

# Set paths.
lightx2v_path=path/to/LightX2V
model_path=path/to/Cosmos3-Nano

# Set environment variables.
export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls cosmos3 \
  --task i2va \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/cosmos3/cosmos3_nano_omni_action_fd_agibotworld_multichunk.json" \
  --prompt "" \
  --image_path "${model_path}/assets/example_action_fd_agibotworld_first_frame.png" \
  --action_path "${model_path}/assets/example_action_fd_agibotworld_action_chunks.json" \
  --save_result_path "${lightx2v_path}/save_results/cosmos3_nano_action_fd_agibotworld_multichunk.mp4" \
  --seed 0
