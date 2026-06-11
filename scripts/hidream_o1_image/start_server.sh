#!/bin/bash

lightx2v_path=/root/yongyang3/LightX2V
model_path=/root/wushuo/models/HiDream-ai/HiDream-O1-Image-Dev-2604
config_json=/root/yongyang3/LightX2V/configs/hidream_o1_image/mlu/hidream_o1_image_t2i_dev_2604_dist.json

host=0.0.0.0
port=8000
metric_port=8001
max_queue_size=10
nproc_per_node=4
master_port=29500

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

echo "Starting HiDream-O1-Image Dev-2604 T2I distributed service on ${host}:${port}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, NPROC_PER_NODE=${nproc_per_node}, MASTER_PORT=${master_port}"

torchrun --nproc_per_node="${nproc_per_node}" --master_port="${master_port}" -m lightx2v.server \
--model_cls hidream_o1_image \
--task t2i \
--model_path "${model_path}" \
--config_json "${config_json}" \
--host "${host}" \
--port "${port}" \
--metric_port "${metric_port}" \
--max_queue_size "${max_queue_size}"

echo "Service stopped"
