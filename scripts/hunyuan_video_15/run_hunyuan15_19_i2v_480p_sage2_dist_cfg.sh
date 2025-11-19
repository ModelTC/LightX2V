#!/bin/bash

# set path and first
lightx2v_path=/data/nvme1/yongyang/LightX2V-HY
model_path=/data/nvme1/models/hy15day0/i2v_new/ckpts/hunyuanvideo-1.5

export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=2 -m lightx2v.infer \
--seed 123 \
--model_cls hunyuan_video_1.5 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hunyuan_video_15/hunyuan_video_15_19_i2v_480p_sage2_dist_cfg.json \
--prompt "镜头跟随猪大哥和猪二哥的背影，他们仓皇地从木屋废墟中向远处的砖屋拼命奔跑。木屋废墟的碎片在地面上微微颤动，砖屋在远处显得越来越近。" \
--negative_prompt "" \
--image_path /data/nvme1/yongyang/hy15_demo/0.jpg \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_hunyuan_video_15_i2v.mp4
