#!/bin/bash

lightx2v_path=/root/yongyang3/LightX2V
model_path=/root/wushuo/models/HiDream-ai/HiDream-O1-Image-Dev-2604

host=0.0.0.0
port=8000

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=4 -m lightx2v.server \
--model_cls hidream_o1_image \
--task t2i \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/hidream_o1_image/mlu/hidream_o1_image_t2i_dev_2604_dist.json" \
--host "${host}" \
--port "${port}"
