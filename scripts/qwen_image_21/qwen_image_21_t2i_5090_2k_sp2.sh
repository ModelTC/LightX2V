#!/bin/bash
set -eo pipefail

# Set the project/model paths here and the checkpoint paths in the JSON config.
lightx2v_path=/path/to/LightX2V
model_path=/path/to/Qwen-Image-2.1
config_json=${CONFIG_JSON:-"${lightx2v_path}/configs/qwen_image_21/qwen_image_21_5090_2k_sp2.json"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=2 -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${config_json}" \
    --prompt "A capybara wearing a wizard hat sits at a desk, reading a book by candlelight. Oil painting style, delicate brushwork, warm tones." \
    --size "${HEIGHT:-2048}" "${WIDTH:-2048}" \
    --seed 42 \
    --save_result_path "${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/qwen_image_21_t2i_5090_2k_sp2.png}"
