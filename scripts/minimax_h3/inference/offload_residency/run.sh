#!/bin/bash

lightx2v_path=/data/liuhongda/lightx2v_offload_opt
model_path="${lightx2v_path}/models/MiniMax-H3"

export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/save_results/minimax_h3_residency/resident.json" \
  --prompt 'integrated_multimodal_description: A cinematic fox walks through a snowy pine forest at dawn. overall_soundscape: Soft wind, crunching snow, and distant birds. non_diegetic_music: Quiet warm strings.' \
  --save_result_path "${lightx2v_path}/save_results/minimax_h3_residency/resident.mp4" \
  --seed 42 "$@"
