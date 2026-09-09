#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/dan/LightX2V
model_path=/data/nvme1/models/MiniMaxAI/MiniMax-H3

# Select one platform. NVIDIA is enabled by default.

# NVIDIA
export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES=0

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
source "${lightx2v_path}/scripts/base/base.sh"

# Supported tasks: fl2av, ref2av
python "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/minimax_h3.json" \
  --task fl2av
