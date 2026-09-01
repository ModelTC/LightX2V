#!/bin/bash
lightx2v_path=/path/to/LightX2V
model_path=/path/to/FLUX.2-klein-base-4B
export CUDA_VISIBLE_DEVICES=5

source ${lightx2v_path}/scripts/base/base.sh

python  -m lightx2v.infer \
    --model_cls flux2 \
    --task i2i \
    --model_path $model_path \
    --prompt "remove the masked foreground object and keep the background unchanged" \
    --image_path "${lightx2v_path}/assets/inputs/inpaint_mask" \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein_i2i_inpaint_mask_cache.png" \
    --config_json "${lightx2v_path}/configs/flux2/flux2_klein_i2i_inpaint_mask_cache.json"
