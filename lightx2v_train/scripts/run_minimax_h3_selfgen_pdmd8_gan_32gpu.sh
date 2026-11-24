#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same command on four ACP nodes, eight GPUs per node.
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
: "${MASTER_ADDR:?Set the same MASTER_ADDR on all four nodes}"
: "${MASTER_PORT:?Set the same MASTER_PORT on all four nodes}"
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_STUDENT_INIT="${H3_STUDENT_INIT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1700_ema_init}"
export H3_SELFGEN_CACHE="${H3_SELFGEN_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16/t2av/metadata.jsonl}"
export H3_PDMD_OUTPUT="${H3_PDMD_OUTPUT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_selfgen_pdmd8_gan_dcm1700_shift6_1000_shift12_2000_20261124}"
export H3_SELFGEN_VIEW="$H3_PDMD_OUTPUT/data/selfgen"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HOME="${HF_HOME:-/tmp/h3_hf_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3_selfgen_pdmd8_pycache}"
if [[ -z "${LOCAL_KERNELS:-}" && -n "${KERNELS_CACHE:-}" ]]; then
    export LOCAL_KERNELS="kernels-community/flash-attn3=$KERNELS_CACHE/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b"
fi
config="$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_selfgen_pdmd8_gan_shift6_12_fsdp32.yaml"
"$H3_PYTHON" "$H3_CODE_ROOT/lightx2v_train/scripts/minimax_h3_selfgen_pdmd4_preflight.py" \
    "$config" --source "$H3_SELFGEN_CACHE" "$@"
command=("$H3_PYTHON" -u -m torch.distributed.run --nnodes=4 --nproc_per_node=8
    "--rdzv_id=${H3_RDZV_ID:-h3_selfgen_pdmd8_gan_shift6_12_20261124}"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" --max_restarts=0
    train.py --config "$config")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$H3_PDMD_OUTPUT/logs"
exec > >(tee -a "$H3_PDMD_OUTPUT/logs/launch.${HOSTNAME:-node}.$(date -u +%Y%m%dT%H%M%SZ).$$.log") 2>&1
cd "$H3_CODE_ROOT/lightx2v_train"
exec "${command[@]}"
