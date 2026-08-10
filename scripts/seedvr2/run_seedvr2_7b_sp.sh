#!/bin/bash

lightx2v_path=${LIGHTX2V_PATH:-/mnt/devsft_afs_2/gushiqiao/LightX2V}
model_path=${MODEL_PATH:-/models/SeedVR2-3B}
video_path=${VIDEO_PATH:-/mnt/devsft_afs_2/gushiqiao/output_lightx2v_minimax_h3_t2av_dmd_lora_4step.mp4}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_seedvr2_7b_sp.mp4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=8 -m lightx2v.infer \
  --model_cls seedvr2 \
  --task sr \
  --sr_ratio 2.0 \
  --video_path "${video_path}" \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/seedvr/seedvr2_7b_sp.json" \
  --save_result_path "${output_path}"
