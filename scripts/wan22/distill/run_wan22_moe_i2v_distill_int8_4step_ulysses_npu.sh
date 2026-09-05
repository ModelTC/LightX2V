#!/bin/bash

# set path firstly
lightx2v_path=/LightX2V
model_path=/Wan2.2-I2V-A14B

if [ -z "$lightx2v_path" ] || [ -z "$model_path" ]; then
    echo "Error: Please set lightx2v_path and model_path in this script before running."
    exit 1
fi

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls wan2.2_moe \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--save_result_path ${lightx2v_path}/save_results/wan_moe_i2v_distill_int8_4step_ulysses_npu.mp4
