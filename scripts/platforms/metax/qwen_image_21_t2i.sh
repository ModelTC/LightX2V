#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/Qwen-Image-2.1

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/metax/qwen_image_21_dmd_4step.json" \
    --prompt 'A coffee shop entrance with a chalkboard reading "Qwen Coffee, $2 per cup" and a neon sign displaying "通义千问".' \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_21_t2i_dmd_4step.png" \
    "$@"
