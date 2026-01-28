#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme0/gushiqiao/models/code/LightX2V
model_path=/data/nvme0/gushiqiao/models/z-image-turbo

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls z_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/z_image/z_image_turbo_t2i_offload.json \
--prompt 'Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.' \
--negative_prompt " " \
--save_result_path ${lightx2v_path}/save_results/z_image_turbo.png \
--seed 42 \
--aspect_ratio "16:9"
