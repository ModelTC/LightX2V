#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/dan/LightX2V
model_path=/data/nvme1/yongyang/nb/models/HiDream-ai/HiDream-O1-Image-Dev-2604

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export SENSITIVE_LAYER_DTYPE=FP32

python -m lightx2v.infer \
--model_cls hidream_o1_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hidream_o1_image/hidream_o1_image_t2i_dev_2604_precise.json \
--prompt "一幅色彩鲜艳、视觉吸引力强的美食节海报设计，采用活泼且富有吸引力的美学风格。顶部区域以鲜艳明快的字体醒目地展示标题：“世界美食节”。标题下方紧邻着用较小字号整齐排列的描述性短语：“一场穿越国际美食的味觉之旅——就在您的社区！”中央区域横向排列着三个清晰的文本重点：“50多个美食摊位”，“现场烹饪演示”和“文化音乐表演”，每个内容都配有极简图标：摊位图标代表美食摊位，厨师帽图标代表烹饪演示，音符图标代表音乐表演。下三分之一区域用清晰易读的字体明确列出活动细节：“2023年10月15日周日，市中心市场广场”。左下角以活泼的草书手写体发出热情邀请：“预留好胃口——带上朋友和家人吧！”" \
--save_result_path ${lightx2v_path}/save_results/hidream_o1_image_t2i_dev_2604_precise.png \
--seed 42
