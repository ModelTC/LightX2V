#!/bin/bash

# System management interface: cnmon

# set path firstly
lightx2v_path=/root/yongyang3/LightX2V
model_path=/root/wushuo/models/Tongyi-MAI/Z-Image-Turbo

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls z_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/platforms/mlu/z_image_turbo_t2i.json \
--prompt 'Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.' \
--negative_prompt " " \
--save_result_path ${lightx2v_path}/save_results/z_image_turbo.png \
--seed 42 \
--aspect_ratio "16:9"
