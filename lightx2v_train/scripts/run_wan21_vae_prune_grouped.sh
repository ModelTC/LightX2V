#!/usr/bin/env bash
set -euo pipefail

component="${1:-}"
phase="${2:-}"
case "${component}:${phase}" in
    encoder:search|encoder:recover|decoder:search|decoder:recover) ;;
    *) echo "Usage: bash $0 encoder|decoder search|recover" >&2; exit 2 ;;
esac

TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TRAIN_ROOT}"
WAN_VAE_GROUPED_PYTHON="${WAN_VAE_GROUPED_PYTHON:-/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python}"
export WAN_VAE_GROUPED_PATH="${WAN_VAE_GROUPED_PATH:-/data/nvme6/gushiqiao/models/Sekotalk-ar-2step/Wan2.1_VAE.pth}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

case "${component}" in
    encoder)
        export WAN_VAE_ENCODER_GROUPED_SEARCH_OUTPUT="${WAN_VAE_ENCODER_GROUPED_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_encoder_prune_search_grouped_keep5_ddp}"
        export WAN_VAE_ENCODER_GROUPED_RECOVER_OUTPUT="${WAN_VAE_ENCODER_GROUPED_RECOVER_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_encoder_prune_recover_grouped_keep5_ddp}"
        export WAN_VAE_ENCODER_GROUPED_SELECTION="${WAN_VAE_ENCODER_GROUPED_SELECTION:-${WAN_VAE_ENCODER_GROUPED_SEARCH_OUTPUT}/export/kept_layers.json}"
        selection="${WAN_VAE_ENCODER_GROUPED_SELECTION}"
        ;;
    decoder)
        export WAN_VAE_DECODER_GROUPED_SEARCH_OUTPUT="${WAN_VAE_DECODER_GROUPED_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_decoder_prune_search_grouped_keep5_ddp}"
        export WAN_VAE_DECODER_GROUPED_RECOVER_OUTPUT="${WAN_VAE_DECODER_GROUPED_RECOVER_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_decoder_prune_recover_grouped_keep5_ddp}"
        export WAN_VAE_DECODER_GROUPED_SELECTION="${WAN_VAE_DECODER_GROUPED_SELECTION:-${WAN_VAE_DECODER_GROUPED_SEARCH_OUTPUT}/export/kept_layers.json}"
        selection="${WAN_VAE_DECODER_GROUPED_SELECTION}"
        ;;
esac

if [[ ! -s "${WAN_VAE_GROUPED_PATH}" ]]; then
    echo "Missing Wan2.1 VAE weights: ${WAN_VAE_GROUPED_PATH}. Set WAN_VAE_GROUPED_PATH to Wan2.1_VAE.pth." >&2
    exit 1
fi
if [[ "${phase}" == "recover" ]]; then
    weights="$(dirname -- "${selection}")/wan21_pruned_${component}.safetensors"
    if [[ ! -s "${selection}" || ! -s "${weights}" ]]; then
        echo "Missing grouped Wan ${component} search export: ${selection} / ${weights}. Finish grouped search first." >&2
        exit 1
    fi
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"

exec "${WAN_VAE_GROUPED_PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node="${#GPU_IDS[@]}" \
    train.py --config "configs/train/vae/wan21_${component}_prune_${phase}_grouped_keep5_ddp.yaml"
