#!/bin/bash

# Set paths.
lightx2v_path=path/to/LightX2V
model_path=path/to/Cosmos3-Nano

# Set environment variables.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=8 -m lightx2v.infer \
  --model_cls cosmos3 \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/cosmos3/cosmos3_nano_omni_t2v_cfg_ulysses_8gpu.json" \
  --prompt "${model_path}/assets/example_t2v_prompt.json" \
  --negative_prompt "${model_path}/assets/negative_prompt.json" \
  --save_result_path "${lightx2v_path}/save_results/cosmos3_nano_omni_t2v_cfg_ulysses_8gpu.mp4" \
  --seed 123
