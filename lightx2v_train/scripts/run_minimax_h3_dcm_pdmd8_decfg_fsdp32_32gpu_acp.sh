#!/usr/bin/env bash
# Run this same task command on each of the four ACP workers.
set -Eeuo pipefail
task="${1:?Pass fl2av or ref2av, optionally followed by --dry-run}"
shift
if [[ ! "$task" =~ ^(fl2av|ref2av)$ || $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: $0 {fl2av|ref2av} [--dry-run]" >&2
    exit 2
fi
: "${MASTER_ADDR:?ACP must set a common MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set a common MASTER_PORT on all four workers}"
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_SELFGEN_CACHE="${H3_SELFGEN_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16}"
export H3_FL2AV_CACHE="${H3_FL2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait}"
export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
adapter_root="${H3_TRAINING_ADAPTER_DIR:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-TrainingAdapter}"
if [[ "$task" == fl2av ]]; then
    subset=t2av
    adapter_file=model_for_comfy_dit.safetensors
else
    subset=ref2av
    adapter_file=model_ref2va_for_comfy_dit.safetensors
fi
export H3_CM_CACHE="${H3_CM_CACHE:-$H3_SELFGEN_CACHE/$subset/metadata.jsonl}"
export H3_NEW_CACHE="${H3_NEW_CACHE:-$H3_SELFGEN_CACHE/$subset/condition_metadata.jsonl}"
export H3_TRAINING_ADAPTER_PATH="${H3_TRAINING_ADAPTER_PATH:-$adapter_root/$adapter_file}"
export H3_TRAINING_ADAPTER_SCALE="${H3_TRAINING_ADAPTER_SCALE:-1.0}"
export H3_STAGED_OUTPUT="${H3_STAGED_OUTPUT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_${task}_dcm1500_pdmd1500000_scoredecfg8_shift12_3_r128a128_adaln16_fake5_32gpu_20261030}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_${task}_dcm1500_pdmd1500000_scoredecfg8_shift12_3_fake5_32gpu_20261030}"
export H3_CM_OUTPUT="$H3_STAGED_OUTPUT/cm/train"
export H3_PDMD_OUTPUT="$H3_STAGED_OUTPUT/pdmd/train"
expected_student_init="$H3_CM_OUTPUT/checkpoint-000001500/student_ema"
if [[ -n "${H3_STUDENT_INIT:-}" && "$H3_STUDENT_INIT" != "$expected_student_init" ]]; then
    echo "H3_STUDENT_INIT must use this task's dCM-1500 EMA: $expected_student_init. Unset the stale value before launching." >&2
    exit 2
fi
export H3_STUDENT_INIT="$expected_student_init"
cm_config="${H3_STAGED_CM_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/turbo/minimax_h3_${task}_dcm1500_decfg8_fsdp32.yaml}"
pdmd_config="${H3_STAGED_PDMD_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_${task}_pdmd1500000_from_cm_decfg8_fsdp32.yaml}"
if [[ -z "${LOCAL_KERNELS:-}" ]]; then
    if [[ -z "${H3_KERNEL_SNAPSHOT:-}" && -n "${KERNELS_CACHE:-}" ]]; then
        H3_KERNEL_SNAPSHOT="$KERNELS_CACHE/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b"
    fi
    if [[ -n "${H3_KERNEL_SNAPSHOT:-}" ]]; then
        [[ -d "$H3_KERNEL_SNAPSHOT" ]] || { echo "Missing FlashAttention-3 snapshot: $H3_KERNEL_SNAPSHOT" >&2; exit 1; }
        export LOCAL_KERNELS="kernels-community/flash-attn3=$H3_KERNEL_SNAPSHOT"
    fi
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train:$H3_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONDONTWRITEBYTECODE=1
command -v "$H3_PYTHON" >/dev/null
cd "$H3_CODE_ROOT/lightx2v_train"
exec "$H3_PYTHON" -u scripts/run_minimax_h3_cm_pdmd_stages.py \
    --task "$task" --cm-config "$cm_config" --pdmd-config "$pdmd_config" "$@"
