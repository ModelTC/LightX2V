#!/usr/bin/env bash
set -Eeuo pipefail

# Run this identical script on all four ACP workers (eight GPUs each).
# Independent DMAD entrypoint: stale H3_CONFIG_PATH/H3_PDMD do not select a recipe.
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_DMAD_CONFIG="${H3_DMAD_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_dmad8_fsdp32.yaml}"
: "${MASTER_ADDR:?ACP must set a common MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set a common MASTER_PORT on all four workers}"
: "${H3_MODEL_PATH:?Set the converted MiniMax-H3 model directory with transformer_ref}"
: "${H3_DMAD_CACHE:?Set the NEW paired DMAD manifest; a condition-only cache is insufficient}"
: "${H3_DMAD_OUTPUT:?Set a NEW shared DMAD output directory}"
export H3_MODEL_PATH H3_DMAD_CACHE H3_DMAD_OUTPUT
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi
command -v "$H3_PYTHON" >/dev/null
if [[ ! -f "$H3_DMAD_CONFIG" ]]; then
    echo "Missing DMAD training config: $H3_DMAD_CONFIG" >&2
    exit 1
fi
H3_DMAD_CONFIG="$(cd "$(dirname "$H3_DMAD_CONFIG")" && pwd)/$(basename "$H3_DMAD_CONFIG")"

# Reuse the offline FlashAttention-3 snapshot from the ACP worker image.
if [[ -z "${H3_KERNEL_SNAPSHOT:-}" && -n "${KERNELS_CACHE:-}" ]]; then
    H3_KERNEL_SNAPSHOT="$KERNELS_CACHE/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b"
fi
if [[ -n "${H3_KERNEL_SNAPSHOT:-}" ]]; then
    [[ -d "$H3_KERNEL_SNAPSHOT" ]] || { echo "Missing FlashAttention-3 snapshot: $H3_KERNEL_SNAPSHOT" >&2; exit 1; }
    export LOCAL_KERNELS="kernels-community/flash-attn3=$H3_KERNEL_SNAPSHOT"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HOME="${HF_HOME:-/tmp/h3_hf_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3_dmad_pycache}"

# No torch import, latent deserialization, CUDA initialization, or data mutation.
"$H3_PYTHON" "$H3_CODE_ROOT/lightx2v_train/scripts/check_minimax_h3_dmad_launch.py" "$H3_DMAD_CONFIG"

cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=4 --nproc_per_node=8
    "--rdzv_id=${H3_RDZV_ID:-h3_ref2av_dmad8_fsdp32}"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$H3_DMAD_CONFIG"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
