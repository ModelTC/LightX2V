#!/usr/bin/env bash
# Run the same task command on all four ACP workers; CM is never launched here.
set -Eeuo pipefail
task="${1:?Pass fl2av or ref2av, optionally followed by --dry-run}"
shift
if [[ ! "$task" =~ ^(fl2av|ref2av)$ || $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: $0 {fl2av|ref2av} [--dry-run]" >&2
    exit 2
fi
: "${MASTER_ADDR:?ACP must provide MASTER_ADDR}"
: "${MASTER_PORT:?ACP must provide MASTER_PORT}"
: "${H3_STUDENT_INIT:?Set the completed CM checkpoint-000001500/student_ema directory for THIS task}"
: "${H3_PDMD_OUTPUT:?Set a new PDMD output directory, separate from the old interrupted PDMD}"
: "${H3_RDZV_ID:?Set the same new run ID on all four workers}"
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_SCORE_MODEL_PATH="${H3_SCORE_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-DeCFG-score}"
export H3_SELFGEN_CACHE="${H3_SELFGEN_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16}"
export H3_FL2AV_CACHE="${H3_FL2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait}"
export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
subset=t2av
[[ "$task" != ref2av ]] || subset=ref2av_imageonly
export H3_NEW_CACHE="${H3_NEW_CACHE:-$H3_SELFGEN_CACHE/$subset/condition_metadata.jsonl}"
export H3_STUDENT_INIT H3_PDMD_OUTPUT H3_RDZV_ID
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
config="${H3_PDMD_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_${task}_pdmd1500000_from_cm_premerged_decfg8_save100_fsdp32.yaml}"
[[ "$config" == /* ]] || config="$H3_CODE_ROOT/$config"
cd "$H3_CODE_ROOT/lightx2v_train"
exec "$H3_PYTHON" -u scripts/run_minimax_h3_pdmd_from_cm_premerged.py --task "$task" --config "$config" "$@"
