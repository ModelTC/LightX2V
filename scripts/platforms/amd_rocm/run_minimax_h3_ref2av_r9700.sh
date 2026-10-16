#!/bin/bash

# MiniMax-H3 Ref2AV on 2 x Radeon AI PRO R9700 (gfx1201, 32 GB), tensor parallel 2.
# Requires an aiter build that provides gfx1201_sage_attention (and gfx1201_sol_attention for sol).
#
# The configs enable "use_adaln_cache": generate the cache once with the same config before inference
# (tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh, --model-variant ref2av).
#
# bash scripts/platforms/amd_rocm/run_minimax_h3_ref2av_r9700.sh [sage|sol]

# set path firstly
lightx2v_path=
model_path=

case "${1:-sage}" in
  sage) config_json=${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_ref2av_r9700_sage.json ;;
  sol) config_json=${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_ref2av_r9700_sol.json ;;
  *) echo "Usage: $0 [sage|sol]" >&2; exit 2 ;;
esac

export PLATFORM=amd_rocm
export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant ref2av \
  --task ref2av \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "Generate an audio-video scene following the references." \
  --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
  --save_result_path "${lightx2v_path}/save_results/minimax_h3_ref2av_r9700_${1:-sage}.mp4" \
  --seed 42
