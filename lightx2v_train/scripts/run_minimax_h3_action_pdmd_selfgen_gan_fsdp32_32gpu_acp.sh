#!/usr/bin/env bash
set -Eeuo pipefail

# Submit this same command on all four ACP workers (eight GPUs per worker).
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
: "${MASTER_ADDR:?ACP must set the same MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set the same MASTER_PORT on all four workers}"
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_GAN_PROMPTS="${H3_GAN_PROMPTS:-/mnt/lm_data_afs/gushiqiao/datasets/data.json}"
export H3_GAN_PROMPT_CACHE="${H3_GAN_PROMPT_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/h3_action_prompts_2000_20261109/condition_cache_124_768p_bf16}"
export H3_GAN_INIT_CHECKPOINT="${H3_GAN_INIT_CHECKPOINT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_pdmd8_from_dcm1700_ema_32gpu_20261029/checkpoint-000005800}"
export H3_SELFGEN_CACHE="${H3_SELFGEN_CACHE:-${H3_GAN_REAL_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16/t2av/metadata.jsonl}}"
export H3_STAGED_OUTPUT="${H3_STAGED_OUTPUT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_action_pdmd100_selfgen_gan_shift12_3_fake5_from5800_32gpu_20261109}"
export H3_ACTION_PDMD_OUTPUT="${H3_ACTION_PDMD_OUTPUT:-$H3_STAGED_OUTPUT/phase1_action_pdmd}"
export H3_SELFGEN_GAN_OUTPUT="${H3_SELFGEN_GAN_OUTPUT:-$H3_STAGED_OUTPUT/phase2_selfgen_gan}"
export H3_ACTION_PDMD_VIEW="${H3_ACTION_PDMD_VIEW:-$H3_STAGED_OUTPUT/data/action_prompts}"
export H3_SELFGEN_VIEW="${H3_SELFGEN_VIEW:-$H3_STAGED_OUTPUT/data/self_generated_t2av}"
export H3_GAN_MAX_ITERS="${H3_GAN_MAX_ITERS:-15000}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_action_pdmd_selfgen_gan_from5800_32gpu_20261109}"
phase1_config="${H3_ACTION_PDMD_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_t2av_action_pdmd8_shift12_3_r128_a128_adaln16_fsdp32.yaml}"
phase2_config="${H3_SELFGEN_GAN_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_t2av_selfgen_pdmd8_gan_shift12_3_r128_a128_adaln16_fsdp32.yaml}"

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
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HOME="${HF_HOME:-/tmp/h3_hf_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3_action_pdmd_selfgen_gan_pycache}"

stages=("$H3_PYTHON" -u "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_action_pdmd_selfgen_gan_stages.py"
    --phase1-config "$phase1_config" --phase2-config "$phase2_config"
    --prompts "$H3_GAN_PROMPTS" --prompt-cache "$H3_GAN_PROMPT_CACHE" --real-cache "$H3_SELFGEN_CACHE")
if [[ "${1:-}" == --dry-run ]]; then
    exec "${stages[@]}" --dry-run
fi
case "${H3_GAN_LAUNCH_LOG:-1}" in
    1)
        mkdir -p "$H3_STAGED_OUTPUT/logs"
        node_name="${HOSTNAME:-$(hostname)}"
        node_name="${node_name//[^a-zA-Z0-9_.-]/_}"
        launch_log="$H3_STAGED_OUTPUT/logs/launch.$node_name.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
        exec > >(tee -a "$launch_log") 2>&1
        echo "[launcher] node=$node_name log=$launch_log"
        ;;
    0) ;;
    *) echo "H3_GAN_LAUNCH_LOG must be 0 or 1" >&2; exit 2 ;;
esac
cd "$H3_CODE_ROOT/lightx2v_train"
exec "${stages[@]}"
