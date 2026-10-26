#!/bin/bash

lightx2v_path=
model_path=

export PLATFORM=mps
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mps/qwen_image_21_viggle_v03.json" \
    --prompt "A capybara wearing a wizard hat, oil painting" \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_t2i_viggle_v03.png"
