#!/usr/bin/env bash
set -euo pipefail
train_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$train_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
python_bin="${H3_PYTHON:-python}"
config="${H3_MOE_CONFIG:-$train_root/configs/train/dense2moe/minimax_h3_t2av_fsdp4.yaml}"
log_dir="${H3_LOG_DIR:-/data/nvme6/gushiqiao/outputs/h3_dense2moe_t2av_cached/logs}"
mkdir -p "$log_dir"
exec 9>"$log_dir/pipeline.lock"
flock -n 9 || { echo "This cache/training pipeline is already running"; exit 1; }
trap 'echo "failed: see current stage log" > "$log_dir/state.txt"' ERR
echo cache > "$log_dir/state.txt"
echo "Caching prompt, video and audio on four GPUs; log: $log_dir/cache.log"
"$python_bin" -m lightx2v_train.trainers.dense2moe.cache --config "$config" --mode prepare >> "$log_dir/cache.log" 2>&1
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m lightx2v_train.trainers.dense2moe.cache --config "$config" --mode encode >> "$log_dir/cache.log" 2>&1
"$python_bin" -m lightx2v_train.trainers.dense2moe.cache --config "$config" --mode finalize >> "$log_dir/cache.log" 2>&1
for stage in calibrate shared moe; do
    echo "$stage" > "$log_dir/state.txt"
    echo "Starting $stage; log: $log_dir/train_$stage.log"
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 \
        -m lightx2v_train.trainers.dense2moe.run --config "$config" --stage "$stage" >> "$log_dir/train_$stage.log" 2>&1
done
echo complete > "$log_dir/state.txt"
echo "Cache and training complete"
