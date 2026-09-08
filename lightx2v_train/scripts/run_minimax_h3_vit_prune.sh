#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TRAIN_ROOT}"
H3_VAE_PYTHON="${H3_VAE_PYTHON:-/data/nvme6/gushiqiao/models/MiniMax-H3/local_diffusers/.venv/bin/python}"
H3_VAE_PARALLEL="${H3_VAE_PARALLEL:-ddp}"
case "${H3_VAE_PARALLEL}" in
    ddp|fsdp) ;;
    *) echo "H3_VAE_PARALLEL must be ddp or fsdp." >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export H3_VAE_PRUNE_SEARCH_OUTPUT="${H3_VAE_PRUNE_SEARCH_OUTPUT:-${TRAIN_ROOT}/output_train/minimax_h3_vit_prune_search_4gpu_${H3_VAE_PARALLEL}}"
export H3_VAE_PRUNE_RECOVER_AUX_3K_OUTPUT="${H3_VAE_PRUNE_RECOVER_AUX_3K_OUTPUT:-${TRAIN_ROOT}/output_train/minimax_h3_vit_prune_recover_aux_3k_4gpu_${H3_VAE_PARALLEL}}"
export H3_VAE_PRUNE_SELECTION="${H3_VAE_PRUNE_SELECTION:-${H3_VAE_PRUNE_SEARCH_OUTPUT}/export/kept_layers.json}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"

run_stage() {
    local stage="$1"
    local config="configs/train/vae/minimax_h3_vit_prune_search_4gpu_${H3_VAE_PARALLEL}.yaml"
    if [[ "${stage}" == "recover" ]]; then
        config="configs/train/vae/minimax_h3_vit_prune_recover_aux_3k_4gpu_${H3_VAE_PARALLEL}.yaml"
    fi
    if [[ "${stage}" == "recover" && ! -s "${H3_VAE_PRUNE_SELECTION}" ]]; then
        echo "Missing layer selection: ${H3_VAE_PRUNE_SELECTION}. Run the search stage first." >&2
        return 1
    fi
    "${H3_VAE_PYTHON}" -m torch.distributed.run \
        --standalone --nproc_per_node="${#GPU_IDS[@]}" \
        train.py --config "${config}"
}

case "${1:-all}" in
    search|recover) run_stage "$1" ;;
    all)
        run_stage search
        run_stage recover
        ;;
    *) echo "Usage: bash $0 [search|recover|all]" >&2; exit 2 ;;
esac
