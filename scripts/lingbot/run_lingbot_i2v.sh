#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme0/gushiqiao/models/tmp/LightX2V
model_path=/data/nvme4/models/lingbot-world-base-act

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls lingbot_world \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/lingbot/lingbot_i2v.json \
--prompt "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls." \
--negative_prompt "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，镜头晃动，画面闪烁，模糊，噪点，水印，签名，文字，变形，" \
--image_path /data/nvme0/gushiqiao/models/lingbot-world/examples/00/image.jpg \
--action_path /data/nvme0/gushiqiao/models/lingbot-world/examples/00 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_lingbot_i2v00.mp4
