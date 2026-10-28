#!/usr/bin/env bash
set -Eeuo pipefail

# Shared Turbo-only launcher. Execute the same command on all four ACP workers.
task="${1:?Pass ref2av or fl2av, optionally followed by --dry-run}"
shift
if [[ ! "$task" =~ ^(ref2av|fl2av)$ || $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: $0 {ref2av|fl2av} [--dry-run]" >&2
    exit 1
fi
: "${MASTER_ADDR:?ACP must set the same MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set the same MASTER_PORT on all four workers}"
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
if [[ "$task" == ref2av ]]; then
    default_config="minimax_h3_ref2av_omni_imageonly_turbo8_shift12_3_r128_a128_adaln16_fsdp32.yaml"
    export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
    export H3_REF2AV_TURBO_OUTPUT="${H3_REF2AV_TURBO_OUTPUT:-$H3_CODE_ROOT/outputs/h3_ref2av_turbo8_shift12_3_r128a128_adaln16_32gpu}"
else
    default_config="minimax_h3_fl2av_3task_turbo8_shift6_3_r128_a128_adaln16_fsdp32.yaml"
    export H3_FL2AV_CACHE="${H3_FL2AV_CACHE:-${H3_MULTITASK_3TASK_PORTRAIT_VIEW:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait}}"
    export H3_FL2AV_TURBO_OUTPUT="${H3_FL2AV_TURBO_OUTPUT:-$H3_CODE_ROOT/outputs/h3_fl2av_3task_turbo8_shift6_3_r128a128_adaln16_32gpu}"
fi
config="${H3_TURBO_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/turbo/$default_config}"
command -v "$H3_PYTHON" >/dev/null
[[ -f "$config" ]] || { echo "Missing H3 Turbo config: $config" >&2; exit 1; }
config="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"

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
export HF_HOME="${HF_HOME:-/tmp/h3_hf_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3_turbo_pycache}"

"$H3_PYTHON" "$H3_CODE_ROOT/lightx2v_train/scripts/minimax_h3_turbo_preflight.py" "$task" "$config"
cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=4 --nproc_per_node=8
    "--rdzv_id=${H3_RDZV_ID:-h3_${task}_turbo8_adaln16_32gpu}"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$config"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
