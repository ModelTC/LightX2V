#!/bin/bash
lightx2v_path=
model_path="/data/temp/FLUX.2-klein-9B"
export CUDA_VISIBLE_DEVICES=0,1

source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=2 -m lightx2v.infer \
    --model_cls flux2_klein \
    --task i2i \
    --model_path $model_path \
    --prompt "A cat dressed like a wizard" \
    --image_path "${lightx2v_path}/save_results/flux2_klein_distill.png"  \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein_i2i_distill_ulysses.png" \
    --config_json "${lightx2v_path}/configs/flux2/flux2_klein_i2i_distill_ulysses.json"
