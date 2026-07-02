#!/bin/bash

# Flux2 Dev T2I Inference Script for MetaX
# System management interface: mx-smi

lightx2v_path=/data/LightX2V
model_path=/data/models/FLUX.2-dev

export PLATFORM="metax_cuda"
export CUDA_VISIBLE_DEVICES=0
export MACA_PATH=${MACA_PATH:-/opt/maca-3.3.0}
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${LD_LIBRARY_PATH}

source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${lightx2v_path}/save_results

python -m lightx2v.infer \
    --model_cls flux2_dev \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/platforms/metax/single/flux2_dev.json \
    --prompt 'Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text "BFL Diffusers" on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom.' \
    --save_result_path ${lightx2v_path}/save_results/flux2_dev_metax.png \
    --seed 42
