#!/bin/bash
lightx2v_path=
model_path="black-forest-labs--FLUX.2-klein-base-4B/snapshots/ppt_260529_30e"
export CUDA_VISIBLE_DEVICES=6

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls flux2_klein \
    --task i2i \
    --model_path $model_path \
    --prompt "remove the masked foreground object and keep the background unchanged" \
    --image_path "${lightx2v_path}/assets/inputs/ppt/img_02.png" \
    --inpaint_mask_path "${lightx2v_path}/assets/inputs/ppt/img_02_mask.png" \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein_i2i_inpaint_mask.png" \
    --config_json "${lightx2v_path}/configs/flux2/flux2_klein_i2i_edit.json"
