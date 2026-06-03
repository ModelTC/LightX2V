#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/wushuo/LightX2V
model_path=/data/nvme1/wushuo/hf_models/HiDream-O1-Image

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls hidream_o1_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/hidream_o1_image/hidream_o1_image_t2i.json \
--prompt "medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room." \
--save_result_path ${lightx2v_path}/save_results/hidream_o1_image_t2i.png \
--seed 32
