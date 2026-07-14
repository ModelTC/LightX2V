#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme5/gushiqiao/codes/LightX2V}
config=${CONFIG:-configs/train/dmd/ltx_t2av_dmd_lora.yaml}

cd "${lightx2v_path}/lightx2v_train"
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,7}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

nproc_per_node=${NPROC_PER_NODE:-2}
if [ "${nproc_per_node}" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="${nproc_per_node}" train.py --config "${config}"
else
    python3 train.py --config "${config}"
fi
