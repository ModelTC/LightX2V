#!/usr/bin/env bash
set -eo pipefail

# Activate an environment with LightX2V and libero_plus dependencies first.
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export WAN_MODEL_PATH="${WAN_MODEL_PATH:-/path/to/Wan2.2-TI2V-5B}"
export CKPT_PATH="${CKPT_PATH:-/path/to/libero/merged_checkpoint.pt}"
export DATASET_STATS_PATH="${DATASET_STATS_PATH:-/path/to/libero/dataset_stats.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DTYPE="${DTYPE:-BF16}" SENSITIVE_LAYER_DTYPE="${SENSITIVE_LAYER_DTYPE:-BF16}"
export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-0}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "${lightx2v_path}"

if [[ -z "${OUT:-}" ]]; then
    mkdir -p evaluate_results/libero_plus
    OUT=$(mktemp -d "${lightx2v_path}/evaluate_results/libero_plus/realtimewam_XXXXXX")
fi
export OUT
mkdir -p "${OUT}"

extra_args=()
if [[ -n "${CONFIG_JSON:-}" ]]; then
    extra_args+=("config_json=${CONFIG_JSON}")
fi
"${PYTHON_BIN:-python}" -u scripts/bench/robotics/run_libero_plus.py \
    "${extra_args[@]}" "$@" 2>&1 | tee "${OUT}/manager.log"
