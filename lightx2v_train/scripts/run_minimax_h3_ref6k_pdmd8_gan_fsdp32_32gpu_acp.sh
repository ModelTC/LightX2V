#!/usr/bin/env bash
# Run the same command on all four ACP nodes, with eight GPUs per node.
set -Eeuo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
: "${MASTER_ADDR:?ACP must provide MASTER_ADDR}"
: "${MASTER_PORT:?ACP must provide MASTER_PORT}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] && (( MASTER_PORT > 0 && MASTER_PORT < 65536 )) || { echo "Invalid MASTER_PORT" >&2; exit 2; }
prepare_port="${H3_REF6K_PREPARE_PORT:-$((MASTER_PORT + 1))}"
[[ "$prepare_port" =~ ^[0-9]+$ ]] && (( prepare_port > 0 && prepare_port < 65536 && prepare_port != MASTER_PORT )) || { echo "H3_REF6K_PREPARE_PORT must be valid and differ from MASTER_PORT" >&2; exit 2; }
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_TEACHER_EXPORT="${H3_TEACHER_EXPORT:-/mnt/devsft_afs_2/gushiqiao/outputs/data_new2/omni_ref_teacher29_base_5k}"
export H3_RAW_JSONL="${H3_RAW_JSONL:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/omni_r2v_image_only_100000_ir.jsonl}"
export H3_MEDIA_ROOT="${H3_MEDIA_ROOT:-/mnt/lm_data_afs/gushiqiao/datasets/omini-r2v}"
export H3_SELFGEN_METADATA="${H3_SELFGEN_METADATA:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16/ref2av_imageonly_768p/metadata.jsonl}"
export H3_REF6K_DATA="${H3_REF6K_DATA:-/mnt/devsft_afs_2/gushiqiao/outputs/data_new2/omni_selfgen_ref6k_base}"
export H3_REF6K_CACHE="$H3_REF6K_DATA/metadata.jsonl"
export H3_STUDENT_INIT="${H3_STUDENT_INIT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_ref2av_imgonly_dcm1500_pdmd1500000_restart_ema999_lr2e5_4e6_20261030/cm/train/checkpoint-000001500/student_ema}"
export H3_REF6K_OUTPUT="${H3_REF6K_OUTPUT:-/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdmd8_gan_cm1500_ref6k_lr5e5_shift12_3_fake5_32gpu_20261116/train}"
export H3_REF6K_MAX_ITERS="${H3_REF6K_MAX_ITERS:-3000}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_ref2av_pdmd8_gan_cm1500_ref6k_20261116}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#devices[@]}" -eq 8 ]] || { echo "Expose eight GPUs per ACP node" >&2; exit 2; }
declare -A seen=()
for device in "${devices[@]}"; do
    [[ -n "$device" && -z "${seen[$device]:-}" ]] || { echo "GPU IDs must be distinct and nonempty" >&2; exit 2; }
    seen[$device]=1
done
if [[ -z "${LOCAL_KERNELS:-}" ]]; then
    if [[ -z "${H3_KERNEL_SNAPSHOT:-}" && -n "${KERNELS_CACHE:-}" ]]; then
        H3_KERNEL_SNAPSHOT="$KERNELS_CACHE/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b"
    fi
    if [[ -n "${H3_KERNEL_SNAPSHOT:-}" ]]; then
        [[ -d "$H3_KERNEL_SNAPSHOT" ]] || { echo "Missing FlashAttention-3 snapshot: $H3_KERNEL_SNAPSHOT" >&2; exit 1; }
        export LOCAL_KERNELS="kernels-community/flash-attn3=$H3_KERNEL_SNAPSHOT"
    fi
fi
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train:$H3_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONDONTWRITEBYTECODE=1
config="${H3_REF6K_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_pdmd8_gan_from_cm1500_ref6k_fsdp32.yaml}"
[[ "$config" == /* ]] || config="$H3_CODE_ROOT/$config"
endpoint="$MASTER_ADDR:$MASTER_PORT"
[[ "$MASTER_ADDR" != *:* || "$MASTER_ADDR" == \[* ]] || endpoint="[$MASTER_ADDR]:$MASTER_PORT"
prepare_endpoint="${endpoint%:*}:$prepare_port"
prepare=("$H3_PYTHON" -u -m torch.distributed.run --nnodes=4 --nproc_per_node=1
    "--rdzv_id=${H3_RDZV_ID}_prepare" --rdzv_backend=c10d "--rdzv_endpoint=$prepare_endpoint" --max_restarts=0
    "$H3_CODE_ROOT/lightx2v_train/scripts/prepare_minimax_h3_ref6k_distributed.py" --config "$config")
train=("$H3_PYTHON" -u -m torch.distributed.run --nnodes=4 --nproc_per_node=8
    "--rdzv_id=$H3_RDZV_ID" --rdzv_backend=c10d "--rdzv_endpoint=$endpoint" --max_restarts=0
    "$H3_CODE_ROOT/lightx2v_train/train.py" --config "$config")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${prepare[@]}"; printf '\n'
    printf '%q ' "${train[@]}"; printf '\n'
    exit 0
fi
mkdir -p "$H3_REF6K_OUTPUT/logs"
node_name="${HOSTNAME:-$(hostname)}"
node_name="${node_name//[^a-zA-Z0-9_.-]/_}"
launch_log="$H3_REF6K_OUTPUT/logs/launch.$node_name.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
exec > >(tee -a "$launch_log") 2>&1
echo "[ref6k-gan] node=$node_name log=$launch_log"
"${prepare[@]}"
printf '[ref6k-gan] '; printf '%q ' "${train[@]}"; printf '\n'
cd "$H3_CODE_ROOT/lightx2v_train"
exec "${train[@]}"
