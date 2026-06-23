#!/bin/bash
lightx2v_path=
model_path="/mnt/miaohua/wangshankun/HF/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/ppt_260529_30e"
export  CUDA_VISIBLE_DEVICES=5,6

source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=2 -m lightx2v.infer \
    --model_cls flux2_klein \
    --task i2i \
    --model_path $model_path \
    --prompt "remove the masked foreground object and keep the background unchanged" \
    --image_path "${lightx2v_path}/assets/inputs/inpaint_mask" \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein_i2i_inpaint_mask_cache.png" \
    --config_json "${lightx2v_path}/configs/flux2/flux2_klein_i2i_inpaint_mask_cfg_parallel_cache.json"
