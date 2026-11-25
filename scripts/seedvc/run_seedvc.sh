#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/models/Seed-VC
data_dir=${lightx2v_path}/../lightx2v_examples/a2a/seedvc
function_id=${1:-1}

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

# Select function 1-6 from the shared data; default: 1.
mapfile -d '' -t sample_args < <(python "${lightx2v_path}/scripts/seedvc/server/sample_utils.py" "$model_path" "$data_dir" "$function_id")

python -m lightx2v.infer \
  --model_cls seedvc \
  --task a2a \
  --config_json ${lightx2v_path}/configs/seedvc/seedvc.json \
  "${sample_args[@]}" \
  --save_result_path ${lightx2v_path}/save_results/seedvc/function_${function_id}.wav
