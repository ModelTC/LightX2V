#!/usr/bin/env bash
set -Eeuo pipefail

train_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${WAN_DMD_MODEL:?Set the official Wan2.1 model directory}"
: "${WAN_DMD_PROMPTS:?Set the training prompts path}"
: "${WAN_DMD_OUTPUT:?Set the experiment output directory}"
: "${CUDA_VISIBLE_DEVICES:?Select two GPUs}"
export WAN_DMD_MODEL WAN_DMD_PROMPTS WAN_DMD_OUTPUT
export WAN_DMD_PROJECTED="${WAN_DMD_PROJECTED:-false}"
export PYTHONPATH="$train_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

cd "$train_root"
printf 'Wan DMD: GPUs=%s projected_dmd=%s output=%s\n' \
    "$CUDA_VISIBLE_DEVICES" "$WAN_DMD_PROJECTED" "$WAN_DMD_OUTPUT"
exec "${WAN_DMD_PYTHON:-python}" -u -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    "--rdzv_id=wan21_${WAN_DMD_PROJECTED}" --max_restarts=0 \
    train.py --config configs/train/dmd/wan2_1_t2v_1_3b_pdmd_comparison_fsdp2.yaml
