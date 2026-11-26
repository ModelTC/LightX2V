#!/usr/bin/env bash
set -euo pipefail
train_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$train_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
python_bin="${H3_PYTHON:-python}"
config="${H3_MOE_CONFIG:-$train_root/configs/train/dense2moe/minimax_h3_t2av_fsdp4.yaml}"
for stage in calibrate shared moe; do
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 \
        -m lightx2v_train.trainers.dense2moe.run --config "$config" --stage "$stage" "$@"
done
