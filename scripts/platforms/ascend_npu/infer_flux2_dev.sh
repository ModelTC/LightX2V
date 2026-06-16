#!/bin/bash

# Flux2 Dev T2I Inference Script for Ascend NPU 910B
# Usage: bash infer_flux2_dev.sh

lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/to5_models/FLUX.2-dev

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${lightx2v_path}/save_results

python -m lightx2v.infer \
    --model_cls flux2_dev \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/platforms/ascend_npu/flux2_dev.json \
    --prompt 'Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text "BFL Diffusers" on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom.' \
    --save_result_path ${lightx2v_path}/save_results/flux2_dev.png \
    --seed 42
