#!/bin/bash
set -e

lightx2v_path=/data/liuhongda/lightx2v_offload_opt
model_path="${lightx2v_path}/models/Wan2.1-I2V-14B-720P-Lightx2v"

export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL=0
cd "${lightx2v_path}"

python -m lightx2v.infer \
  --model_cls wan2.1 \
  --task i2v \
  --model_path "${model_path}" \
  --config_json "save_results/wan_residency/resident.json" \
  --prompt "A white cat wearing sunglasses sits on a surfboard on a sunny beach." \
  --negative_prompt "过曝，静态，模糊，字幕，低质量，畸形" \
  --image_path "assets/inputs/imgs/img_0.jpg" \
  --save_result_path "save_results/wan_residency/resident.mp4" \
  --seed 42
