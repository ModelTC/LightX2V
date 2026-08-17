#!/bin/bash
set -e

# set path firstly
lightx2v_path=/data/nvme0/gushiqiao/codes/LightX2V
model_path=/data/nvme0/gushiqiao/models/LTX-2.5

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls ltx2_5 \
--task t2av \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/ltx2/ltx2_5_distilled_8gpu.json" \
--prompt "A cinematic fox walking through a snowy forest." \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/ltx2_5_distilled_t2av.mp4" \
