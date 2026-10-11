#!/usr/bin/env bash

# AdaLN cache setup:
# If the inference JSON config enables "use_adaln_cache": true, generate the cache before inference:
# 1. Set lightx2v_path, model_path, --config_json, and --model-variant in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh.
# 2. Use --model-variant fl2av for t2av/i2av/l2av/fl2av, or --model-variant ref2av for ref2av.
# 3. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
# Cache generation and inference must use the same JSON config and adaln_cache_dir.
set -eo pipefail

# Set paths and the LoRA checkpoint path in the inference JSON's lora_configs.
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export ZE_AFFINITY_MASK=0
export PLATFORM=intel_xpu
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

source ${lightx2v_path}/scripts/base/base.sh
export PYTHONPATH=${lightx2v_path}/lightx2v_kernel_xpu/python:${PYTHONPATH}
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

torchrun --standalone --nproc_per_node=1 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path ${model_path} \
  --config_json ${lightx2v_path}/configs/platforms/intel_xpu/minimax_h3_t2av_dmd_lora_4step.json \
  --prompt "A cinematic fox walking through a snowy forest, with soft wind and distant birds." \
  --save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_dmd_lora_4step.mp4 \
  --seed 42
