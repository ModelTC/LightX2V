#!/bin/bash
set -e

lightx2v_path=/data/liuhongda/lightx2v_offload_opt
model_path="${lightx2v_path}/models/MiniMax-H3"

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL=0
cd "${lightx2v_path}"

python -m tools.estimate_offload_residency \
  --model-cls minimax_h3 \
  --model-variant fl2av \
  --task t2av \
  --model-path "${model_path}" \
  --config "configs/minimax_h3/residency/baseline_npu.json" \
  --prompt 'integrated_multimodal_description: A cinematic fox walks through a snowy pine forest at dawn. overall_soundscape: Soft wind, crunching snow, and distant birds. non_diegetic_music: Quiet warm strings.' \
  --output-config "save_results/minimax_h3_residency/resident_npu.json" "$@"
