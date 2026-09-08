#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TRAIN_ROOT}"
H3_VAE_PYTHON="${H3_VAE_PYTHON:-/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export H3_VAE_ENCODER_SEARCH_OUTPUT="${H3_VAE_ENCODER_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/minimax_h3_encoder_prune_search_keep3_2gpu_ddp}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne 2 ]]; then
    echo "Encoder search requires two GPUs; set GPU_LIST, for example 6,7." >&2
    exit 2
fi

exec "${H3_VAE_PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    train.py --config configs/train/vae/minimax_h3_encoder_prune_search_keep3_2gpu_ddp.yaml
