#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TRAIN_ROOT}"
WAN_VAE_PYTHON="${WAN_VAE_PYTHON:-/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/data/nvme6/gushiqiao/models/Sekotalk-ar-2step/Wan2.1_VAE.pth}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-6,7}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

component="${1:-}"
case "${component}" in
    encoder)
        keep=3
        export WAN_VAE_ENCODER_SEARCH_OUTPUT="${WAN_VAE_ENCODER_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_encoder_prune_search_keep3_ddp}"
        export WAN_VAE_ENCODER_RECOVER_OUTPUT="${WAN_VAE_ENCODER_RECOVER_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_encoder_prune_recover_keep3_ddp}"
        export WAN_VAE_ENCODER_SELECTION="${WAN_VAE_ENCODER_SELECTION:-${WAN_VAE_ENCODER_SEARCH_OUTPUT}/export/kept_layers.json}"
        selection="${WAN_VAE_ENCODER_SELECTION}"
        ;;
    decoder)
        keep=5
        export WAN_VAE_DECODER_SEARCH_OUTPUT="${WAN_VAE_DECODER_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_decoder_prune_search_keep5_ddp}"
        export WAN_VAE_DECODER_RECOVER_OUTPUT="${WAN_VAE_DECODER_RECOVER_OUTPUT:-${TRAIN_ROOT}/output_train/wan21_decoder_prune_recover_keep5_ddp}"
        export WAN_VAE_DECODER_SELECTION="${WAN_VAE_DECODER_SELECTION:-${WAN_VAE_DECODER_SEARCH_OUTPUT}/export/kept_layers.json}"
        selection="${WAN_VAE_DECODER_SELECTION}"
        ;;
    *) echo "Usage: bash $0 encoder|decoder [search|recover|all]" >&2; exit 2 ;;
esac

if [[ ! -s "${WAN_VAE_PATH}" ]]; then
    echo "Missing Wan2.1 VAE weights: ${WAN_VAE_PATH}. Set WAN_VAE_PATH to Wan2.1_VAE.pth." >&2
    exit 1
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"

run_stage() {
    local stage="$1"
    if [[ "${stage}" == "recover" ]]; then
        local weights="$(dirname -- "${selection}")/wan21_pruned_${component}.safetensors"
        if [[ ! -s "${selection}" || ! -s "${weights}" ]]; then
            echo "Missing Wan ${component} search export: ${selection} / ${weights}. Finish its search stage first." >&2
            return 1
        fi
    fi
    "${WAN_VAE_PYTHON}" -m torch.distributed.run \
        --standalone --nproc_per_node="${#GPU_IDS[@]}" \
        train.py --config "configs/train/vae/wan21_${component}_prune_${stage}_keep${keep}_ddp.yaml"
}

case "${2:-all}" in
    search|recover) run_stage "$2" ;;
    all)
        run_stage search
        run_stage recover
        ;;
    *) echo "Usage: bash $0 encoder|decoder [search|recover|all]" >&2; exit 2 ;;
esac
