#!/bin/bash
lightx2v_path=
model_path="/data/temp/black-forest-labs/FLUX.2-klein-9B"
export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls flux2_klein \
    --task i2i \
    --model_path $model_path \
    --prompt "图1的人物穿上图2中所有的服饰,并保持人物的姿势不变" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/img_merge"  \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein_i2i_distill_img_merge.png" \
    --config_json "${lightx2v_path}/configs/flux2/flux2_klein_i2i_distill.json"
