#!/bin/bash
# Qwen Image i2i example with PyTorch trace profiling enabled.
# Docs: docs/ZH_CN/source/method_tutorials/torch_profiling.md

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export lightx2v_path="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export model_path="${MODEL_PATH:-/data/nvme1/models/qwen-image-edit-release-251130}"

export LIGHTX2V_TORCH_PROFILE=1
export LIGHTX2V_TORCH_PROFILE_FORMAT=both
export LIGHTX2V_TORCH_PROFILE_TB_DIR=${lightx2v_path}/save_results/torch_profile
export LIGHTX2V_TORCH_PROFILE_CHROME=${lightx2v_path}/save_results/trace_magi.json
export TENSORBOARD_PORT=16006

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/qwen_image/qwen_image_i2i_2511.json \
    --prompt "Change the person to a standing position, bending over to hold the dog's front paws." \
    --negative_prompt " " \
    --image_path "${lightx2v_path}/save_results/img1.png" \
    --save_result_path ${lightx2v_path}/save_results/qwen_image_i2i_2511.png \
    --seed 0
