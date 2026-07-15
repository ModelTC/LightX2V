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
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path /data/nvme5/gushiqiao/cases/wecom-temp-3950334-bfa56035a08485356431b5a1c5c28a82.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/infinitetalk_single_720p_dist_8gpus.mp4 \
--seed 42
