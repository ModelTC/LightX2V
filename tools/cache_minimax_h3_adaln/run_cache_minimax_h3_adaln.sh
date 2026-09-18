#!/bin/bash
set -eo pipefail

# set path firstly
lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MINIMAX_H3_MODEL_PATH:-${lightx2v_path}/models/MiniMax-H3}"
cache_task="${MINIMAX_H3_CACHE_TASK:-fl2av}"

# Select one platform. NVIDIA is enabled by default.

# NVIDIA
export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

# Supported model variants: fl2av, ref2av
exec python -m tools.cache_minimax_h3_adaln.cache_minimax_h3_adaln \
  --model_path "${model_path}" \
  --config_json "${MINIMAX_H3_CONFIG:-${lightx2v_path}/configs/minimax_h3/minimax_h3.json}" \
  --model-variant "${cache_task}"
