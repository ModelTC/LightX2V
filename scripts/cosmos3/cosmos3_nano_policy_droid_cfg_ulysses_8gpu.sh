#!/bin/bash

# Set paths.
lightx2v_path=path/to/LightX2V
model_path=path/to/Cosmos3-Nano-Policy-DROID

# Set environment variables.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=8 -m lightx2v.infer \
  --model_cls cosmos3 \
  --task i2va \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/cosmos3/cosmos3_nano_policy_droid_cfg_ulysses_8gpu.json" \
  --prompt "Pick up the banana and place it in the bowl" \
  --image_path "path/to/banana_in_bowl/observation.png" \
  --state_path "path/to/banana_in_bowl/state.npy" \
  --save_action_path "${lightx2v_path}/save_results/cosmos3_nano_policy_droid_action.npy" \
  --seed 0
