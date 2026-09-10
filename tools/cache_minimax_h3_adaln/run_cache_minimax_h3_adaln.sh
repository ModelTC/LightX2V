#!/bin/bash
set -euo pipefail

# set path firstly
lightx2v_path=/Users/yongyang/Documents/x2v/LightX2V
model_path=/Users/yongyang/Documents/x2v/models/MiniMaxAI/diffusers/MiniMax-H3
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/mps/minimax_h3_t2av_4step_512_22.json}"

# Select one platform. Apple MPS is enabled by default.

# Apple MPS
export PLATFORM=mps

# NVIDIA
# export PLATFORM=cuda
# export CUDA_VISIBLE_DEVICES=0

# Intel XPU
# export PLATFORM=intel_xpu
# export ZE_AFFINITY_MASK=0

# AMD ROCm
# export PLATFORM=amd_rocm
# export CUDA_VISIBLE_DEVICES=0

# MetaX
# export PLATFORM=metax_cuda
# export CUDA_VISIBLE_DEVICES=0

# Ascend NPU
# export PLATFORM=ascend_npu
# export ASCEND_RT_VISIBLE_DEVICES=0

# MThreads MUSA
# export PLATFORM=musa
# export MUSA_VISIBLE_DEVICES=0

# Cambricon MLU
# export PLATFORM=cambricon_mlu
# export MLU_VISIBLE_DEVICES=0

# Hygon DCU
# export PLATFORM=hygon_dcu
# export HIP_VISIBLE_DEVICES=0

# Enflame GCU
# export PLATFORM=enflame_gcu
# export ECCL_RAS_DISABLE=2

# Iluvatar CoreX
# export PLATFORM=iluvatar_cuda
# export CUDA_VISIBLE_DEVICES=0

# PPU
# export PLATFORM=ppu_cuda
# export CUDA_VISIBLE_DEVICES=0

# set environment variables
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"

# fl2av also generates the cache used by t2av/i2av/l2av inference.
# Cache generation and inference must use the same model and JSON config.
/opt/miniconda3/envs/torch/bin/python "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --task fl2av
