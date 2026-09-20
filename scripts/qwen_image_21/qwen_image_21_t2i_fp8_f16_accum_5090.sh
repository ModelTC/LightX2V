#!/bin/bash
set -eo pipefail

# Set the project/model paths here and dit_quantized_ckpt in the JSON config.
lightx2v_path=/path/to/LightX2V
model_path=/path/to/Qwen-Image-2.1

export CUDA_VISIBLE_DEVICES=0
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/qwen_image_21/qwen_image_21_t2i_fp8_f16_accum_5090.json" \
    --prompt "A capybara wearing a wizard hat sits at a desk, reading a book by candlelight. Oil painting style, delicate brushwork, warm tones." \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_t2i_fp8_f16_accum.png"
