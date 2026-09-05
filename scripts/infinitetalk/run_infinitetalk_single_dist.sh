#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme4/gushiqiao/new/debug/LightX2V
model_path=/data/nvme5/gushiqiao/models/InfiniteTalk

export CUDA_VISIBLE_DEVICES=7


# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls infinitetalk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/infinitetalk/5090/infinitetalk_single_distilled_8gpus.json \
--prompt  "让角色根据音频内容自然说话" \
--image_path /data/nvme5/gushiqiao/cases/wecom-temp-3950334-bfa56035a08485356431b5a1c5c28a82.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/infinitetalk_single_720p_dist_8gpus.mp4 \
--seed 42
