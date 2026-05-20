#!/bin/bash
lightx2v_path=
model_path="/data/temp/black-forest-labs/FLUX.2-klein-9B"
export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls flux2_klein \
    --task i2i \
    --model_path $model_path \
    --prompt "A clean ppt background, without text" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/mask_edit"  \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein_i2i_mask_edit.png" \
    --config_json "${lightx2v_path}/configs/flux2/flux2_klein_i2i_edit.json"
