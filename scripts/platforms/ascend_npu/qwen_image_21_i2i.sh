#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Qwen-Image-2.1
example_path=/data/test_images/edit/08_背包穿戴与遮挡

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export ASCEND_HOME_PATH="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/ascend_npu/qwen_image_21_dmd_4step.json" \
    --prompt "$(cat "${example_path}/prompt.txt")" \
    --image_path "${example_path}/图1.png,${example_path}/图2.png,${example_path}/图3.png" \
    --size 928 1120 \
    --seed 49 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_i2i_dmd_4step.png" \
    "$@"
