#!/bin/bash
lightx2v_path=/mnt/afs1/wangshankun/LightX2V
model_path="/data/temp/FLUX.2-klein-9B"
export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

python -m debugpy --listen 0.0.0.0:15678 -m lightx2v.infer \
    --model_cls flux2_klein \
    --task t2i \
    --model_path $model_path \
    --prompt "A cat holding a sign that says hello world" \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein_distill.png" \
    --config_json "${lightx2v_path}/configs/flux2_klein/flux2_klein_distill.json"
