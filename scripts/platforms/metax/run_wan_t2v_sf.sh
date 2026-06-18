#!/bin/bash

# Self-Forcing Wan2.1 T2V Inference Script for MetaX
# System management interface: mx-smi

lightx2v_path=/data/LightX2V
model_path=/data/models/Wan2.1-T2V-1.3B

export PLATFORM="metax_cuda"
export CUDA_VISIBLE_DEVICES=0
export MACA_PATH=${MACA_PATH:-/opt/maca-3.3.0}
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${LD_LIBRARY_PATH}

source ${lightx2v_path}/scripts/platforms/metax/logging.sh

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1_sf \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/platforms/metax/wan_t2v_sf.json \
--prompt 'A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides.' \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan_t2v_sf_metax.mp4
