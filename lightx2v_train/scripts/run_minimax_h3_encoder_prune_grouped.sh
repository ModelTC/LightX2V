#!/usr/bin/env bash
set -euo pipefail

phase="${1:-}"
case "${phase}" in
    search|recover) ;;
    *) echo "Usage: bash $0 search|recover" >&2; exit 2 ;;
esac

TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TRAIN_ROOT}"
H3_VAE_GROUPED_PYTHON="${H3_VAE_GROUPED_PYTHON:-/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export H3_VAE_ENCODER_GROUPED_SEARCH_OUTPUT="${H3_VAE_ENCODER_GROUPED_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/minimax_h3_encoder_prune_search_grouped_keep6_2gpu_ddp}"
export H3_VAE_ENCODER_GROUPED_RECOVER_OUTPUT="${H3_VAE_ENCODER_GROUPED_RECOVER_OUTPUT:-${TRAIN_ROOT}/output_train/minimax_h3_encoder_prune_recover_grouped_keep6_2gpu_ddp}"
export H3_VAE_ENCODER_GROUPED_SELECTION="${H3_VAE_ENCODER_GROUPED_SELECTION:-${H3_VAE_ENCODER_GROUPED_SEARCH_OUTPUT}/export/kept_layers.json}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"

if [[ "${phase}" == "recover" ]]; then
    weights="$(dirname -- "${H3_VAE_ENCODER_GROUPED_SELECTION}")/minimax_h3_pruned_encoder.safetensors"
    if [[ ! -s "${H3_VAE_ENCODER_GROUPED_SELECTION}" || ! -s "${weights}" ]]; then
        echo "Missing grouped H3 search export: ${H3_VAE_ENCODER_GROUPED_SELECTION} / ${weights}. Finish grouped search first." >&2
        exit 1
    fi
fi

exec "${H3_VAE_GROUPED_PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node="${#GPU_IDS[@]}" \
    train.py --config "configs/train/vae/minimax_h3_encoder_prune_${phase}_grouped_keep6_2gpu_ddp.yaml"
