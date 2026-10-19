#!/bin/bash

lightx2v_path=/data/liuhongda/lightx2v_offload_opt
model_path="${lightx2v_path}/models/Qwen-Image-2.1-Turbo"

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls qwen_image_21 \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/qwen_image_21/residency/baseline_npu.json" \
  --prompt 'A capybara wearing a wizard hat sits at a desk, reading a book by candlelight. Oil painting style, delicate brushwork, warm tones.' \
  --save_result_path "${lightx2v_path}/save_results/qwen_image_21_residency/baseline_npu.png" \
  --seed 42 "$@"
