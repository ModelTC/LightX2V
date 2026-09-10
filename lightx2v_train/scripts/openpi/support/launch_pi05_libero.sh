#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
train_root="$(cd -- "${script_dir}/../../.." && pwd)"
lightx2v_root="$(dirname -- "${train_root}")"
workspace_root="$(dirname -- "${lightx2v_root}")"
openpi_data_root="${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}"
runtime="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-${openpi_data_root}/python_deps/openpi_official_pytorch_runtime}"
runtime_tool="${lightx2v_root}/scripts/openpi/support/runtime.py"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export OPENPI_INITIAL_CHECKPOINT="${OPENPI_INITIAL_CHECKPOINT:-${openpi_data_root}/openpi-assets/checkpoints/pi05_base_pytorch_fp32}"
export OPENPI_LEROBOT_ROOT="${OPENPI_LEROBOT_ROOT:-${openpi_data_root}/lerobot/physical-intelligence/libero}"
export OPENPI_NORM_STATS_PATH="${OPENPI_NORM_STATS_PATH:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json}"
export OPENPI_TRAIN_OUTPUT="${OPENPI_TRAIN_OUTPUT:-${lightx2v_root}/output_train/openpi/pi05_libero}"
export OPENPI_TRANSFORMERS_RUNTIME_PATH="${runtime}"
export OPENPI_GLOBAL_BATCH_SIZE="${OPENPI_GLOBAL_BATCH_SIZE:-256}"
export OPENPI_GRADIENT_ACCUMULATION_ITERS="${OPENPI_GRADIENT_ACCUMULATION_ITERS:-1}"
export HF_HOME="${HF_HOME:-${openpi_data_root}/hf_cache}"
export USE_FLAX=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nproc="${OPENPI_NPROC_PER_NODE:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}"
if [[ ! "${nproc}" =~ ^[1-9][0-9]*$ || ! "${OPENPI_GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ || ! "${OPENPI_GRADIENT_ACCUMULATION_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU count, global batch size, and gradient accumulation must be positive integers." >&2
  exit 2
fi

batch_divisor=$((nproc * OPENPI_GRADIENT_ACCUMULATION_ITERS))
if ((OPENPI_GLOBAL_BATCH_SIZE % batch_divisor != 0)); then
  echo "OPENPI_GLOBAL_BATCH_SIZE must be divisible by GPUs * gradient accumulation." >&2
  exit 2
fi
per_gpu_batch=$((OPENPI_GLOBAL_BATCH_SIZE / batch_divisor))

python "${runtime_tool}" train-check
echo "OpenPI train: GPUs=${nproc}, per-GPU batch=${per_gpu_batch}, accumulation=${OPENPI_GRADIENT_ACCUMULATION_ITERS}, global batch=${OPENPI_GLOBAL_BATCH_SIZE}"
if [[ -n "${OPENPI_RESUME_CHECKPOINT:-}" ]]; then
  echo "Initialization: full resume from ${OPENPI_RESUME_CHECKPOINT}"
else
  echo "Initialization: model-only warm start from ${OPENPI_INITIAL_CHECKPOINT}"
fi

export PYTHONPATH="${runtime}:${lightx2v_root}:${train_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${train_root}"
exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${nproc}" \
  train.py \
  --config configs/train/openpi/pi05_libero.yaml
