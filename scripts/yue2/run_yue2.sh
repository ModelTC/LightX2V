#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/models/YuE2-3B
data_dir=${lightx2v_path}/assets/inputs/yue2
function_id=${1:-1}

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

# Select function 1-13 from the shared data; default: 1.
mapfile -d '' -t sample_args < <(python "${lightx2v_path}/scripts/yue2/server/sample_utils.py" "$data_dir" "$function_id")

python -m lightx2v.infer \
  --model_cls yue2 \
  --task t2a \
  --model_path "$model_path" \
  --config_json ${lightx2v_path}/configs/yue2/yue2.json \
  "${sample_args[@]}" \
  --save_result_path ${lightx2v_path}/save_results/yue2/function_${function_id}.flac
