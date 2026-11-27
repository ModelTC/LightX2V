#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Usage: bash scripts/run_h3_manifest_acp.sh pdmd32|pdmd_dmad32|moe16 [--dry-run]}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export H3_MODEL_PATH=${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}
export H3_MANIFEST=${H3_MANIFEST:-/mnt/miaohua/gushiqiao/datasets/fl2av_dataset/h3_self_gen/dataset.json}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [[ -n ${KERNELS_CACHE:-} && -z ${LOCAL_KERNELS:-} ]]; then
    export LOCAL_KERNELS="kernels-community/flash-attn3=$KERNELS_CACHE/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b"
fi
case "$mode" in
    pdmd32|pdmd_dmad32)
        nodes=4
        config=configs/train/dmd/minimax_h3_manifest_pdmd4_fsdp32.yaml
        export H3_STUDENT_INIT=${H3_STUDENT_INIT:-/mnt/lm_data_afs/gushiqiao/outputs/h3_fl2av_dcm1700_ema_init}
        run=h3_manifest_pdmd4_gan
        if [[ $mode == pdmd_dmad32 ]]; then
            config=configs/train/dmd/minimax_h3_manifest_pdmd4_dmad_fsdp32.yaml
            run=h3_fl2av_pdmd4_dmad_single_shift12_audio3_ratio1
        fi
        export H3_PDMD_OUTPUT=${H3_PDMD_OUTPUT:-/mnt/miaohua/gushiqiao/outputs/$run}
        output=$H3_PDMD_OUTPUT
        ;;
    moe16)
        nodes=2
        config=configs/train/dense2moe/minimax_h3_manifest_fsdp16.yaml
        export H3_TEXT_MODEL_PATH=${H3_TEXT_MODEL_PATH:-$H3_MODEL_PATH}
        export H3_MOE_OUTPUT=${H3_MOE_OUTPUT:-/mnt/miaohua/gushiqiao/outputs/h3_manifest_dense2moe}
        output=$H3_MOE_OUTPUT
        ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

if [[ ${2:-} != --dry-run ]]; then
    mkdir -p "$output/logs"
    exec > >(tee -a "$output/logs/${HOSTNAME}.log") 2>&1
fi
launch() {
    local stage=$1
    shift
    local cmd=("${H3_PYTHON:-python}" -m torch.distributed.run --nnodes="$nodes" --nproc_per_node=8
        --rdzv_backend=c10d "--rdzv_endpoint=${MASTER_ADDR:?}:${MASTER_PORT:?}"
        "--rdzv_id=${H3_RDZV_ID:-h3_manifest_$mode}_$stage" --max_restarts=0 "$@")
    if [[ $dry_run == --dry-run ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
}
dry_run=${2:-}
if [[ $mode != moe16 ]]; then
    previous=configs/train/dmd/minimax_h3_selfgen_pdmd4_gan_shift6_12_fsdp32.yaml
    if [[ $mode == pdmd_dmad32 ]]; then previous=configs/train/dmd/minimax_h3_manifest_pdmd4_fsdp32.yaml; fi
    "${H3_PYTHON:-python}" - "$config" "$previous" <<'PY'
import sys
from omegaconf import OmegaConf

old = OmegaConf.load(sys.argv[2]).training
new = OmegaConf.load(sys.argv[1]).training
for label, role in (("student", "student"), ("fake", "fake"), ("GAN head", "fake")):
    print(f"{label} LR: {old[role].optimizer.learning_rate:.6f} -> {new[role].optimizer.learning_rate:.6f}", flush=True)
PY
    launch pdmd train.py --config "$config"
else
    for stage in calibrate shared moe; do
        if [[ $stage == moe ]]; then
            launch empty_text data_process/minimax_h3/cache_inverse_cfg_conditions.py --config "$config"
        fi
        launch "$stage" -m lightx2v_train.trainers.dense2moe.run --config "$config" --stage "$stage"
    done
fi
