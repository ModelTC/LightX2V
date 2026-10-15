#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.2-T2V-A14B

export CUDA_VISIBLE_DEVICES=0,1,2,3
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=4 -m lightx2v.server \
  --model_cls wan2.2_moe \
  --task t2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/wan22/wan_moe_t2v_tp.json \
  --host 0.0.0.0 \
  --port 8000
