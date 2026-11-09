#!/usr/bin/env bash
set -Eeuo pipefail

# Run this same command on all four ACP workers (eight GPUs per worker).
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
: "${MASTER_ADDR:?ACP must set the same MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set the same MASTER_PORT on all four workers}"
: "${H3_GAN_REAL_CACHE:?Set a real T2AV AV-latent metadata.jsonl or cache directory, including its matching condition paths}"
if [[ ! "$MASTER_PORT" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
    echo "MASTER_PORT must be in 1..65535" >&2; exit 2
fi
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_GAN_PROMPTS="${H3_GAN_PROMPTS:-/mnt/lm_data_afs/gushiqiao/datasets/data.json}"
export H3_GAN_PROMPT_CACHE="${H3_GAN_PROMPT_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/h3_action_prompts_2000_20261109/condition_cache_124_768p_bf16}"
export H3_GAN_INIT_CHECKPOINT="${H3_GAN_INIT_CHECKPOINT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_pdmd8_from_dcm1700_ema_32gpu_20261029/checkpoint-000005800}"
export H3_GAN_OUTPUT="${H3_GAN_OUTPUT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_t2av_action_pdmd8_gan_shift12_3_fake5_from5800_32gpu_20261109}"
export H3_GAN_MAX_ITERS="${H3_GAN_MAX_ITERS:-15000}"
export H3_GAN_REAL_CACHE
config="${H3_GAN_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_t2av_action_pdmd8_gan_shift12_3_r128_a128_adaln16_fsdp32.yaml}"
preprocess_port="${H3_GAN_PREPROCESS_PORT:-$((MASTER_PORT + 1))}"
if [[ ! "$preprocess_port" =~ ^[0-9]+$ ]] || (( preprocess_port < 1 || preprocess_port > 65535 || preprocess_port == MASTER_PORT )); then
    echo "H3_GAN_PREPROCESS_PORT must be in 1..65535 and differ from MASTER_PORT" >&2; exit 2
fi
rdzv_id="${H3_RDZV_ID:-h3_t2av_action_pdmd8_gan_from5800_32gpu_20261109}"

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
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#devices[@]}" -eq 8 ]] || { echo "Each ACP worker must expose exactly eight GPUs" >&2; exit 2; }
declare -A seen=()
for device in "${devices[@]}"; do
    [[ -n "$device" && -z "${seen[$device]:-}" ]] || { echo "GPU IDs must be nonempty and unique" >&2; exit 2; }
    seen[$device]=1
done
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HOME="${HF_HOME:-/tmp/h3_hf_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3_pdmd_gan_pycache}"

preflight=("$H3_PYTHON" "$H3_CODE_ROOT/lightx2v_train/scripts/minimax_h3_pdmd_gan_preflight.py" "$config" --prompts "$H3_GAN_PROMPTS")
preprocess=("$H3_PYTHON" -u -m torch.distributed.run --nnodes=4 --nproc_per_node=8
    "--rdzv_id=$rdzv_id-prompts" --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$preprocess_port" --max_restarts=0
    "$H3_CODE_ROOT/lightx2v_train/data_process/minimax_h3/prepare_minimax_h3_gan_prompts.py"
    --prompts "$H3_GAN_PROMPTS" --model-path "$H3_MODEL_PATH" --output-dir "$H3_GAN_PROMPT_CACHE"
    --expected-samples 2000 --expected-world-size 32)
train=("$H3_PYTHON" -u -m torch.distributed.run --nnodes=4 --nproc_per_node=8
    "--rdzv_id=$rdzv_id-train" --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" --max_restarts=0
    "$H3_CODE_ROOT/lightx2v_train/train.py" --config "$config")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${preflight[@]}"; printf '\n'
    printf '%q ' "${preprocess[@]}"; printf '\n'
    printf '%q ' "${preflight[@]}" --check-cache; printf '\n'
    printf '%q ' "${train[@]}"; printf '\n'
    exit 0
fi
case "${H3_GAN_LAUNCH_LOG:-1}" in
    1)
        mkdir -p "$H3_GAN_OUTPUT/logs"
        node_name="${HOSTNAME:-$(hostname)}"
        node_name="${node_name//[^a-zA-Z0-9_.-]/_}"
        launch_log="$H3_GAN_OUTPUT/logs/launch.$node_name.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
        exec > >(tee -a "$launch_log") 2>&1
        echo "[launcher] node=$node_name log=$launch_log"
        ;;
    0) ;;
    *) echo "H3_GAN_LAUNCH_LOG must be 0 or 1" >&2; exit 2 ;;
esac
"${preflight[@]}"
"${preprocess[@]}"
"${preflight[@]}" --check-cache
cd "$H3_CODE_ROOT/lightx2v_train"
exec "${train[@]}"
