#!/usr/bin/env bash
set -euo pipefail

task=${1:?Choose fl2av or ref2av}
mode=${2:-}
steps=${H3_PDMD_STEPS:-8}
train_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dcm_config=$train_dir/configs/train/consistency/minimax_h3_${task}_dcm_fsdp32.yaml
pdmd_config=$train_dir/configs/train/pdmd/minimax_h3_${task}_pdmd${steps}_gan_fsdp32.yaml
if [[ $# -gt 2 || ( -n $mode && $mode != --dry-run ) || ! -f $dcm_config || ! -f $pdmd_config ]]; then
    echo "Usage: $0 <fl2av|ref2av> [--dry-run] (H3_PDMD_STEPS=4 or 8)" >&2
    exit 2
fi

: "${H3_MODEL_PATH:?Set the MiniMax-H3 model directory}"
: "${H3_TRAIN_META:?Set the shared meta.json path}"
: "${H3_OUTPUT_DIR:?Set the shared output directory}"

export PLATFORM=nvidia
export PYTHONPATH="$train_dir:${train_dir%/*}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export H3_DCM_OUTPUT="$H3_OUTPUT_DIR/dcm" H3_PDMD_OUTPUT="$H3_OUTPUT_DIR/pdmd${steps}"
export H3_DCM_TRAIN_ITERS=${H3_DCM_TRAIN_ITERS:-1500}
printf -v checkpoint '%s/checkpoint-%09d' "$H3_DCM_OUTPUT" "$H3_DCM_TRAIN_ITERS"
export H3_STUDENT_INIT="$checkpoint/student_ema"

nnodes=${NNODES:-4}
nproc=${NPROC_PER_NODE:-8}
rank=${NODE_RANK:?Set the node rank}
port=${MASTER_PORT:-29500}
pdmd_port=${H3_PDMD_PORT:-$((port + 1))}
if (( nproc < 1 || nnodes * nproc != 32 || rank < 0 || rank >= nnodes ||
      port < 1 || port > 65535 || pdmd_port < 1 || pdmd_port > 65535 || port == pdmd_port )); then
    echo "Use 32 processes, a valid node rank, and distinct ports for dCM and PDMD." >&2
    exit 2
fi
launch=(torchrun --nnodes="$nnodes" --nproc-per-node="$nproc"
    --node-rank="$rank" --master-addr="${MASTER_ADDR:?Set the rank 0 node address}")
dcm=("${launch[@]}" --master-port="$port" "$train_dir/train.py" --config "$dcm_config")
pdmd=("${launch[@]}" --master-port="$pdmd_port" "$train_dir/train.py" --config "$pdmd_config")

if [[ $mode == --dry-run ]]; then
    printf '%q ' "${dcm[@]}"
    printf '\n'
    printf '%q ' "${pdmd[@]}"
    printf '\n'
    exit 0
fi

weights=$H3_STUDENT_INIT/pytorch_lora_weights.safetensors
if [[ ! -s $weights || ! -f $checkpoint/dist_state/.metadata ]]; then
    "${dcm[@]}"
fi
test -s "$weights"
exec "${pdmd[@]}"
