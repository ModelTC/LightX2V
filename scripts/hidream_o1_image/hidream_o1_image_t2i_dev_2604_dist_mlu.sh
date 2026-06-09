#!/bin/bash

# System management interface: cnmon

# set path firstly
lightx2v_path=/root/yongyang3/LightX2V
model_path=/root/wushuo/models/HiDream-ai/HiDream-O1-Image-Dev-2604

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=4 -m lightx2v.infer \
--model_cls hidream_o1_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hidream_o1_image/mlu/hidream_o1_image_t2i_dev_2604_dist.json \
--prompt "medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room." \
--save_result_path ${lightx2v_path}/save_results/hidream_o1_image_t2i_dev_2604_dist_mlu.png \
--seed 32
