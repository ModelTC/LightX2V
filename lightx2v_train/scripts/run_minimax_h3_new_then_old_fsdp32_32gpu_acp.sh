#!/usr/bin/env bash
set -Eeuo pipefail

task="${1:?Pass fl2av or ref2av, then dmd or pdmd}"
method="${2:?Pass dmd or pdmd}"
shift 2
if [[ ! "$task" =~ ^(fl2av|ref2av)$ || ! "$method" =~ ^(dmd|pdmd)$ || $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: $0 {fl2av|ref2av} {dmd|pdmd} [--dry-run]" >&2
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
export H3_STUDENT_INIT="${H3_STUDENT_INIT:-}"
export H3_TRAINING_ADAPTER_ENABLE="${H3_TRAINING_ADAPTER_ENABLE:-false}"
export H3_TRAINING_ADAPTER_SCALE="${H3_TRAINING_ADAPTER_SCALE:-1.0}"
case "$H3_TRAINING_ADAPTER_ENABLE" in
    1|true|True) H3_TRAINING_ADAPTER_ENABLE=true ;;
    0|false|False) H3_TRAINING_ADAPTER_ENABLE=false ;;
    *) echo "H3_TRAINING_ADAPTER_ENABLE must be true/false or 1/0." >&2; exit 2 ;;
esac
adapter_tag=ta0
[[ "$H3_TRAINING_ADAPTER_ENABLE" == false ]] || adapter_tag=ta1_scoredecfg
adapter_root="${H3_TRAINING_ADAPTER_DIR:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-TrainingAdapter}"
if [[ "$task" == fl2av ]]; then
    export H3_NEW_CACHE="${H3_NEW_CACHE:-$H3_SELFGEN_CACHE/t2av/condition_metadata.jsonl}"
    export H3_TRAINING_ADAPTER_PATH="${H3_TRAINING_ADAPTER_PATH:-$adapter_root/model_for_comfy_dit.safetensors}"
else
    export H3_NEW_CACHE="${H3_NEW_CACHE:-$H3_SELFGEN_CACHE/ref2av/condition_metadata.jsonl}"
    export H3_TRAINING_ADAPTER_PATH="${H3_TRAINING_ADAPTER_PATH:-$adapter_root/model_ref2va_for_comfy_dit.safetensors}"
fi
export H3_CURRICULUM_OUTPUT="${H3_CURRICULUM_OUTPUT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_${task}_${method}8_new_then_old_${adapter_tag}_fake5_32gpu_20261029}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_${task}_${method}8_new_then_old_${adapter_tag}_fake5_32gpu_20261029}"
config="${H3_CURRICULUM_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_${task}_${method}8_new_then_old_adaln16_fsdp32.yaml}"
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
"$H3_PYTHON" "$H3_CODE_ROOT/lightx2v_train/scripts/minimax_h3_curriculum_preflight.py" "$task" "$method" "$config"
command=("$H3_PYTHON" -u -m torch.distributed.run --nnodes=4 --nproc_per_node=8
    "--rdzv_id=$H3_RDZV_ID" --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$config")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
cd "$H3_CODE_ROOT/lightx2v_train"
exec "${command[@]}"
