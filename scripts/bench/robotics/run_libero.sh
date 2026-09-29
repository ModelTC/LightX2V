#!/bin/bash

set -eo pipefail

# Set paths first; activate the matching benchmark environment before running.
lightx2v_path="${LIGHTX2V_PATH:-/path/to/LightX2V}"
model_path="${WAN_MODEL_PATH:-/path/to/Wan2.2-TI2V-5B}"
base_ckpt="${BASE_CKPT:-/path/to/libero/base_checkpoint.pt}"
lora_path="${LORA_PATH:-/path/to/libero/checkpoint-000030000}"
dataset_stats_path="${DATASET_STATS_PATH:-/path/to/libero/dataset_stats.json}"
python_bin="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONPATH="${PYTHONPATH:-}"
# Select the repository-local benchmark; explicit CLI overrides remain supported.
unset LIBERO_SOURCE_DIR LIBERO_PLUS_SOURCE_DIR
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

source "${lightx2v_path}/scripts/base/base.sh"
cd "${lightx2v_path}"

output_dir="${OUT:-}"
if [[ -z "${output_dir}" ]]; then
    mkdir -p "${lightx2v_path}/evaluate_results/libero"
    output_dir=$(mktemp -d "${lightx2v_path}/evaluate_results/libero/realtimewam_teacher_ema30000_1step_XXXXXX")
fi
mkdir -p "${output_dir}"

"${python_bin}" -u scripts/bench/robotics/run_libero.py \
    model=realtimewam \
    model.backbone=fasterwam \
    model.sampler=teacher_flow \
    model.lora_weights=ema \
    base_ckpt="${base_ckpt}" \
    lora_path="${lora_path}" \
    ckpt=null \
    model.model_path="${model_path}" \
    EVALUATION.dataset_stats_path="${dataset_stats_path}" \
    EVALUATION.num_inference_steps=1 \
    EVALUATION.action_infer_mode=one_pass_future_cache \
    EVALUATION.sigma_shift=5.0 \
    EVALUATION.replan_steps=10 \
    EVALUATION.num_trials=50 \
    MULTIRUN.num_gpus=8 \
    MULTIRUN.max_tasks_per_gpu=2 \
    'MULTIRUN.task_suite_names=[libero_10,libero_goal,libero_spatial,libero_object]' \
    seed=42 \
    EVALUATION.output_dir="${output_dir}" \
    "$@" \
    2>&1 | tee "${output_dir}/manager.log"
