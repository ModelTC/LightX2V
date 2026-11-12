#!/usr/bin/env bash
set -Eeuo pipefail

task="${1:?Pass fl2av or ref2av, optionally followed by --dry-run}"
shift
if [[ ! "$task" =~ ^(fl2av|ref2av)$ || $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: $0 {fl2av|ref2av} [--dry-run]" >&2
    exit 2
fi
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_PDD_CONFIG="${H3_PDD_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/pdd/minimax_h3_${task}_pdd8_fsdp32.yaml}"
export H3_PDD_OUTPUT="${H3_PDD_OUTPUT:-/mnt/devsft_afs_2/gushiqiao/outputs/h3_${task}_pdd8_32gpu}"
export H3_PDD_MAX_ITERS="${H3_PDD_MAX_ITERS:-30000}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_${task}_pdd8_32gpu}"
if [[ "$task" == fl2av ]]; then
    export H3_PDD_CACHE="${H3_PDD_CACHE:-${H3_FL2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait}}"
else
    export H3_PDD_CACHE="${H3_PDD_CACHE:-${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ "${1:-}" == --dry-run ]]; then
    address="${MASTER_ADDR:-ACP_MASTER_ADDR}"
    port="${MASTER_PORT:-ACP_MASTER_PORT}"
else
    : "${MASTER_ADDR:?ACP must set the same MASTER_ADDR on all four workers}"
    : "${MASTER_PORT:?ACP must set the same MASTER_PORT on all four workers}"
    address="$MASTER_ADDR"
    port="$MASTER_PORT"
    [[ -e "$H3_PDD_CACHE" ]] || { echo "Missing H3_PDD_CACHE: $H3_PDD_CACHE" >&2; exit 1; }
    [[ -f "$H3_PDD_CONFIG" ]] || { echo "Missing PDD config: $H3_PDD_CONFIG" >&2; exit 1; }
    if [[ -z "${LOCAL_KERNELS:-}" && -n "${KERNELS_CACHE:-}" ]]; then
        snapshot="${H3_KERNEL_SNAPSHOT:-$KERNELS_CACHE/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b}"
        [[ -d "$snapshot" ]] || { echo "Missing FlashAttention-3 snapshot: $snapshot" >&2; exit 1; }
        export LOCAL_KERNELS="kernels-community/flash-attn3=$snapshot"
    fi
    mkdir -p "$H3_PDD_OUTPUT/logs"
    launch_log="$H3_PDD_OUTPUT/logs/launch.$(hostname).$(date -u +%Y%m%dT%H%M%SZ).$$.log"
    exec > >(tee -a "$launch_log") 2>&1
    echo "[h3-pdd] launcher log: $launch_log"
fi

command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=4 --nproc_per_node=8
    "--rdzv_id=$H3_RDZV_ID" --rdzv_backend=c10d
    "--rdzv_endpoint=$address:$port" --max_restarts=0
    train.py --config "$H3_PDD_CONFIG"
)
echo "H3 $task: PDD, 4 x 8 GPUs, 8 steps, shifts12/3, no fake model."
echo "data=$H3_PDD_CACHE output=$H3_PDD_OUTPUT/train student_updates=$H3_PDD_MAX_ITERS"
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${1:-}" == --dry-run ]]; then
    exit 0
fi
cd "$H3_CODE_ROOT/lightx2v_train"
exec "${command[@]}"
