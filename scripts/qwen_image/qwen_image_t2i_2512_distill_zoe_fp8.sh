#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/ccc/LightX2V
model_path=/data/nvme1/models/Qwen/Qwen-Image-2512

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls qwen_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/qwen_image/qwen_image_t2i_2512_distill_zoe_fp8.json \
--prompt '2K超高清画质，16:9宽屏比例，电影级渲染。一个精致的咖啡店门口场景，温馨的街道氛围。门口摆放着一个复古风格的木质黑板，黑板上用粉笔字体写着"日日新咖啡，2美元一杯"，笔触温馨可爱。旁边有一个闪烁的霓虹灯招牌，红色霓虹灯管拼出"商汤科技"字样，现代科技感。旁边立着一幅精美的海报，海报上是一位优雅的中国美女模特，海报下方用时尚字体写着"SenseNova newbee"。整体氛围是东西方文化交融的现代咖啡馆，暖色调灯光，傍晚时分，细节精致，高质量渲染' \
--negative_prompt " " \
--save_result_path ${lightx2v_path}/save_results/qwen_image_t2i_2512_distill_zoe_fp83.png \
--seed 42 \
--target_shape 1536 2752
