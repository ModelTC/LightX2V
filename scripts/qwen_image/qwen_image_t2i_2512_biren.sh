#!/bin/bash

# Set these paths before running, or export LIGHTX2V_PATH and MODEL_PATH.
lightx2v_path=${LIGHTX2V_PATH:-/path/to/LightX2V}
model_path=${MODEL_PATH:-/path/to/Qwen-Image-2512}

export PLATFORM=biren_supa

# Optional: select cards without putting a hardware-specific card list in the
# application script. The Biren plugin maps this to SUPA_VISIBLE_DEVICES.
# export BIREN_VISIBLE_DEVICES=4

source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=1 -m lightx2v.infer \
  --model_cls qwen_image \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/qwen_image/qwen_image_t2i_2512_biren.json" \
  --prompt "a cup of coffee" \
  --save_result_path "${lightx2v_path}/save_results/qwen_image_t2i_2512_biren.png" \
  --seed 42
