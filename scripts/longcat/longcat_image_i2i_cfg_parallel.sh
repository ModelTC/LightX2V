#!/bin/bash

export lightx2v_path=/workspace
export model_path=/data/nvme1/models/meituan-longcat/LongCat-Image-Edit

export CUDA_VISIBLE_DEVICES=0,1

# Source base configuration if exists
if [ -f "${lightx2v_path}/scripts/base/base.sh" ]; then
    source ${lightx2v_path}/scripts/base/base.sh
fi

torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls longcat_image \
--task i2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/longcat_image/longcat_image_i2i_cfg_parallel.json \
--prompt "将猫变成狗" \
--negative_prompt "" \
--image_path /data/nvme1/models/meituan-longcat/LongCat-Image-Edit/assets/test.png \
--save_result_path ${lightx2v_path}/save_results/longcat_image_i2i_cfg_parallel.png \
--seed 43
