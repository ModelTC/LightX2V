#!/usr/bin/env bash
set -Eeuo pipefail

train_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${WAN_DMD_MODEL:?Set the official Wan2.1 model directory}"
: "${WAN_DMD_PROMPTS:?Set the training prompts path}"
: "${WAN_DMD_OUTPUT:?Set a fresh experiment output directory}"
: "${CUDA_VISIBLE_DEVICES:?Select two GPUs}"
export WAN_DMD_MODEL WAN_DMD_PROMPTS WAN_DMD_OUTPUT
export WAN_DMD_PROJECTED="${WAN_DMD_PROJECTED:-false}"
export WAN_DMD_RESIDUAL_HEAD="${WAN_DMD_RESIDUAL_HEAD:-false}"
if [[ "$WAN_DMD_PROJECTED" == true && "$WAN_DMD_RESIDUAL_HEAD" == true ]]; then
    printf 'Select projected DMD or residual-head DMD, not both.\n' >&2
    exit 2
fi
export PYTHONPATH="$train_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

cd "$train_root"
printf 'Wan 4-step DMD: GPUs=%s projected=%s residual_head=%s output=%s\n' \
    "$CUDA_VISIBLE_DEVICES" "$WAN_DMD_PROJECTED" "$WAN_DMD_RESIDUAL_HEAD" "$WAN_DMD_OUTPUT"
exec "${WAN_DMD_PYTHON:-python}" -u -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    "--rdzv_id=wan21_4step_${WAN_DMD_PROJECTED}_${WAN_DMD_RESIDUAL_HEAD}" \
    --max_restarts=0 train.py \
    --config configs/train/dmd/wan2_1_t2v_1_3b_head_comparison_fsdp2.yaml
