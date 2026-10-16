#!/bin/bash
set -e

lightx2v_path=/data/liuhongda/lightx2v_offload_opt
model_path="${lightx2v_path}/models/Wan2.1-I2V-14B-720P-Lightx2v"

export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL=0
cd "${lightx2v_path}"

python -m tools.estimate_offload_residency \
  --task i2v \
  --model-path "${model_path}" \
  --config "configs/wan/residency/baseline.json" \
  --image-path "assets/inputs/imgs/img_0.jpg" \
  --output-config "save_results/wan_residency/resident.json"
