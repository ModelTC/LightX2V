#!/bin/bash

lightx2v_path=/data/liuhongda/lightx2v_offload_opt
model_path="${lightx2v_path}/models/Qwen-Image-2.1-Turbo"

export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m tools.estimate_offload_residency \
  --model-cls qwen_image_21 \
  --task t2i \
  --model-path "${model_path}" \
  --config "${lightx2v_path}/configs/qwen_image_21/residency/baseline.json" \
  --prompt 'A capybara wearing a wizard hat sits at a desk, reading a book by candlelight. Oil painting style, delicate brushwork, warm tones.' \
  --output-config "${lightx2v_path}/save_results/qwen_image_21_residency/resident.json" "$@"
