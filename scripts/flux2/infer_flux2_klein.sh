#!/bin/bash
lightx2v_path=/path/to/LightX2V
model_path=/path/to/FLUX.2-klein-base-9B
export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls flux2 \
    --task t2i \
    --target_shape 1024 1024 \
    --model_path $model_path \
    --prompt "A cat holding a sign that says hello world" \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein.png" \
    --config_json "${lightx2v_path}/configs/flux2/flux2_klein.json"
