#!/bin/bash

# Set paths.
lightx2v_path=path/to/LightX2V
model_path=path/to/BAGEL-7B-MoT

# Set environment variables.
export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls bagel \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/bagel/bagel_t2i_cot.json" \
  --prompt "A female cosplayer portraying an ethereal fairy or elf in a magical forest with glowing plants." \
  --aspect_ratio 1:1 \
  --seed 42 \
  --save_result_path "${lightx2v_path}/save_results/bagel_t2i_cot.png"
