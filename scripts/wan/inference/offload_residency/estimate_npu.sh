#!/bin/bash

lightx2v_path=/data/liuhongda/lightx2v_offload_opt
model_path="/path/to/Wan2.1-I2V-14B-720P"

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m tools.estimate_offload_residency \
  --task i2v \
  --model-path "${model_path}" \
  --config "${lightx2v_path}/configs/wan/residency/baseline_npu.json" \
  --image-path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
  --output-config "${lightx2v_path}/save_results/wan_residency/resident_npu.json"
