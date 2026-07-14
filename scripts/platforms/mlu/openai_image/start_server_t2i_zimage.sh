#!/bin/bash

# set path firstly
lightx2v_path=/root/yongyang3/LightX2V
model_path=/root/wushuo/models/Tongyi-MAI/Z-Image-Turbo

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0,1
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Start API server with distributed inference service
torchrun --nproc_per_node=2 -m lightx2v.server \
--model_cls z_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/platforms/mlu/z_image_turbo_t2i_dist.json \
--port 8000

echo "Service stopped"
