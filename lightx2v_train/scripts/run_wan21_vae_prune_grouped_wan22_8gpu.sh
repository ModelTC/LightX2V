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
WAN_VAE_GROUPED_PYTHON="${WAN_VAE_GROUPED_PYTHON:-python}"
export WAN_VAE_GROUPED_PATH="${WAN_VAE_GROUPED_PATH:-/mnt/devsft_afs_1/gushiqiao/Wan2.1_VAE.pth}"
export WAN_VAE_METADATA="${WAN_VAE_METADATA:-${TRAIN_ROOT}/data/wan22/metadata_vae.jsonl}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

case "${component}" in
    encoder)
        export WAN_VAE_ENCODER_WAN22_SEARCH_OUTPUT="${WAN_VAE_ENCODER_WAN22_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_encoder_prune_search_grouped_keep5_wan22_8gpu_ddp}"
        export WAN_VAE_ENCODER_WAN22_RECOVER_OUTPUT="${WAN_VAE_ENCODER_WAN22_RECOVER_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_encoder_prune_recover_grouped_keep5_wan22_8gpu_ddp}"
        export WAN_VAE_ENCODER_WAN22_SELECTION="${WAN_VAE_ENCODER_WAN22_SELECTION:-${WAN_VAE_ENCODER_WAN22_SEARCH_OUTPUT}/export/kept_layers.json}"
        selection="${WAN_VAE_ENCODER_WAN22_SELECTION}"
        ;;
    decoder)
        export WAN_VAE_DECODER_WAN22_SEARCH_OUTPUT="${WAN_VAE_DECODER_WAN22_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_decoder_prune_search_grouped_keep5_wan22_8gpu_ddp}"
        export WAN_VAE_DECODER_WAN22_RECOVER_OUTPUT="${WAN_VAE_DECODER_WAN22_RECOVER_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_decoder_prune_recover_grouped_keep5_wan22_8gpu_ddp}"
        export WAN_VAE_DECODER_WAN22_SELECTION="${WAN_VAE_DECODER_WAN22_SELECTION:-${WAN_VAE_DECODER_WAN22_SEARCH_OUTPUT}/export/kept_layers.json}"
        selection="${WAN_VAE_DECODER_WAN22_SELECTION}"
        ;;
esac

if [[ ! -s "${WAN_VAE_GROUPED_PATH}" ]]; then
    echo "Missing Wan2.1 VAE weights: ${WAN_VAE_GROUPED_PATH}." >&2
    exit 1
fi
if [[ ! -s "${WAN_VAE_METADATA}" ]]; then
    echo "Missing cleaned video metadata: ${WAN_VAE_METADATA}. Copy metadata_vae.jsonl first." >&2
    exit 1
fi
if [[ "${phase}" == "recover" ]]; then
    weights="$(dirname -- "${selection}")/wan21_pruned_${component}.safetensors"
    if [[ ! -s "${selection}" || ! -s "${weights}" ]]; then
        echo "Missing Wan22 grouped ${component} search export: ${selection} / ${weights}. Finish its search stage first." >&2
        exit 1
    fi
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"

exec "${WAN_VAE_GROUPED_PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node="${#GPU_IDS[@]}" \
    train.py --config "configs/train/vae/wan21_${component}_prune_${phase}_grouped_keep5_wan22_8gpu_ddp.yaml"
