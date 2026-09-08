#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls wan2.2_moe \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/distill/wan22/wan_moe_i2v_distill_lora_4step_cfg_ulysses.json \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--save_result_path ${lightx2v_path}/save_results/wan_moe_i2v_distill_lora_4step_cfg_ulysses.mp4
