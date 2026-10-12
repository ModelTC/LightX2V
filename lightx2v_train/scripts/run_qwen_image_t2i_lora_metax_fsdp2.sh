#!/usr/bin/env bash
set -euo pipefail

export MACA_PATH="${MACA_PATH:-/opt/maca-3.7.1}"
export PATH="${MACA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PLATFORM=metax_cuda
# Default to two devices and one worker per visible device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES//,/ }"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#CUDA_DEVICES[@]}}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || (( NPROC_PER_NODE > ${#CUDA_DEVICES[@]} )); then
    printf 'NPROC_PER_NODE must be a positive integer no greater than the number of CUDA_VISIBLE_DEVICES.\n' >&2
    exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-configs/train/flow/qwen_image_lora_dist.yaml}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py \
    --config "${CONFIG_PATH}" \
    "$@"
