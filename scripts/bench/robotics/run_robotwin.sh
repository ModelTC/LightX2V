#!/bin/bash

set -eo pipefail

# Set paths first; activate the matching benchmark environment before running.
lightx2v_path="${LIGHTX2V_PATH:-/path/to/LightX2V}"
model_path="${WAN_MODEL_PATH:-/path/to/Wan2.2-TI2V-5B}"
base_ckpt="${BASE_CKPT:-/path/to/robotwin/base_checkpoint.pt}"
lora_path="${LORA_PATH:-/path/to/robotwin/checkpoint-000030000}"
dataset_stats_path="${DATASET_STATS_PATH:-/path/to/robotwin/dataset_stats.json}"
python_bin="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONPATH="${PYTHONPATH:-}"
# Use the repository-local benchmark and its seed cache by default.
unset ROBOTWIN_ROOT ROBOTWIN_SEED_DIR

source "${lightx2v_path}/scripts/base/base.sh"
cd "${lightx2v_path}"

output_dir="${OUT:-}"
if [[ -z "${output_dir}" ]]; then
    mkdir -p "${lightx2v_path}/evaluate_results/robotwin"
    output_dir=$(mktemp -d "${lightx2v_path}/evaluate_results/robotwin/realtimewam_teacher_ema30000_1step_XXXXXX")
fi
mkdir -p "${output_dir}"

"${python_bin}" -u scripts/bench/robotics/run_robotwin.py \
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
    EVALUATION.replan_steps=8 \
    EVALUATION.skip_get_obs_within_replan=true \
    EVALUATION.eval_num_episodes=100 \
    EVALUATION.reuse_seed_cache=true \
    MULTIRUN.num_gpus=8 \
    MULTIRUN.max_tasks_per_gpu=2 \
    'MULTIRUN.phases=[clean,random]' \
    seed=42 \
    EVALUATION.output_dir="${output_dir}" \
    "$@" \
    2>&1 | tee "${output_dir}/manager.log"
