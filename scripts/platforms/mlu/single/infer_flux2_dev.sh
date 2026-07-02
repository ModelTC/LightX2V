#!/bin/bash

# Flux2 Dev T2I Inference Script for Cambricon MLU
# Usage: bash infer_flux2_dev.sh

# set path firstly
lightx2v_path=/data/LightX2V
model_path=/data/models/FLUX.2-dev

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}


source ${lightx2v_path}/scripts/platforms/mlu/logging.sh

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${lightx2v_path}/save_results

python -m lightx2v.infer \
    --model_cls flux2_dev \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/platforms/mlu/single/flux2_dev.json \
    --prompt 'Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text "BFL Diffusers" on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom.' \
    --save_result_path ${lightx2v_path}/save_results/flux2_dev_mlu.png \
    --seed 42
