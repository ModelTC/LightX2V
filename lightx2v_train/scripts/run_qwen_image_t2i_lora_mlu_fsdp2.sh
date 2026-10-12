#!/usr/bin/env bash
set -euo pipefail

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_MLU_ALLOC_CONF="${PYTORCH_MLU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

read -r -a MLU_DEVICES <<< "${MLU_VISIBLE_DEVICES//,/ }"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#MLU_DEVICES[@]}}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || (( NPROC_PER_NODE > ${#MLU_DEVICES[@]} )); then
    printf 'NPROC_PER_NODE must be a positive integer no greater than the number of MLU_VISIBLE_DEVICES.\n' >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

if (( $# == 0 )); then
    set -- --config configs/train/flow/qwen_image_lora_dist.yaml
fi

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py \
    "$@"
