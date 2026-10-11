#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/InfiniteTalk

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls infinitetalk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/infinitetalk/5090/infinitetalk_single_distilled_8gpus.json \
--prompt  "让角色根据音频内容自然说话" \
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/infinitetalk_single_720p_dist_8gpus.mp4 \
--seed 42
