#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.1-T2V-14B-StepDistill-CfgDistill

export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
  --model_cls wan2.1 \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/distill/wan_t2v_distill_4step_cfg_dynamic.json \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --target_shape 480 832 \
  --seed 42 \
  --save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan_t2v_distill_4step_cfg_dynamic.mp4
