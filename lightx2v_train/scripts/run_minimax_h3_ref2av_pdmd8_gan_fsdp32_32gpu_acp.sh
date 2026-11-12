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
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_SCORE_MODEL_PATH="${H3_SCORE_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-DeCFG-score}"
export H3_SELFGEN_CACHE="${H3_SELFGEN_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16}"
export H3_REF_GAN_REAL_CACHE="${H3_REF_GAN_REAL_CACHE:-$H3_SELFGEN_CACHE/ref2av_imageonly/metadata.jsonl}"
export H3_REF_GAN_INIT_CHECKPOINT="${H3_REF_GAN_INIT_CHECKPOINT:-/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdmd8_from_cm1500_decfg_shift12_3_fake10_save100_32gpu_20261103/train/checkpoint-000005700}"
export H3_REF_GAN_OUTPUT="${H3_REF_GAN_OUTPUT:-/mnt/devsft_afs_2/gushiqiao/outputs/h3_ref2av_pdmd8_gan_from5700_shift12_3_fake5_32gpu_20261112/train}"
export H3_REF_GAN_MAX_ITERS="${H3_REF_GAN_MAX_ITERS:-15000}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_ref2av_pdmd8_gan_from5700_shift12_3_fake5_32gpu_20261112}"
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
config="${H3_REF_GAN_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_pdmd8_gan_from5700_shift12_3_fake5_fsdp32.yaml}"
[[ "$config" == /* ]] || config="$H3_CODE_ROOT/$config"
endpoint="$MASTER_ADDR:$MASTER_PORT"
[[ "$MASTER_ADDR" != *:* || "$MASTER_ADDR" == \[* ]] || endpoint="[$MASTER_ADDR]:$MASTER_PORT"
preflight=("$H3_PYTHON" -u "$H3_CODE_ROOT/lightx2v_train/scripts/minimax_h3_ref2av_pdmd_gan_preflight.py" "$config")
train=("$H3_PYTHON" -u -m torch.distributed.run --nnodes=4 --nproc_per_node=8
    "--rdzv_id=$H3_RDZV_ID" --rdzv_backend=c10d "--rdzv_endpoint=$endpoint" --max_restarts=0
    "$H3_CODE_ROOT/lightx2v_train/train.py" --config "$config")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${preflight[@]}"; printf '\n'
    printf '%q ' "${train[@]}"; printf '\n'
    exit 0
fi
mkdir -p "$H3_REF_GAN_OUTPUT/logs"
node_name="${HOSTNAME:-$(hostname)}"
node_name="${node_name//[^a-zA-Z0-9_.-]/_}"
launch_log="$H3_REF_GAN_OUTPUT/logs/launch.$node_name.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
exec > >(tee -a "$launch_log") 2>&1
echo "[ref-gan] node=$node_name log=$launch_log"
if [[ "$H3_REF_GAN_REAL_CACHE" == "$H3_SELFGEN_CACHE/ref2av_imageonly/metadata.jsonl" && ! -f "${H3_REF_GAN_REAL_CACHE%.jsonl}.complete.json" ]]; then
    "$H3_PYTHON" -u "$H3_CODE_ROOT/lightx2v_train/data_process/minimax_h3/prepare_modelscope_h3_dataset.py" imageonly-view --output-dir "$H3_SELFGEN_CACHE"
fi
"${preflight[@]}"
printf '[ref-gan] '; printf '%q ' "${train[@]}"; printf '\n'
cd "$H3_CODE_ROOT/lightx2v_train"
exec "${train[@]}"
