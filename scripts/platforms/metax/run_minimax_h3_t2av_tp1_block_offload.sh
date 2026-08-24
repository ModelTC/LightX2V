#!/bin/bash

lightx2v_path=${LIGHTX2V_PATH:-/data/LightX2V}
model_path=${MINIMAX_H3_MODEL_PATH:-/data/models/MiniMax-H3}
save_result_path=${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_tp1_block_offload.mp4}

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH=/opt/conda/bin:${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/lightx2v-minimax-h3-tp1-block-offload}

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

prompt='In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.'

torchrun --standalone --nproc_per_node=1 -m lightx2v.infer \
    --model_cls minimax_h3 \
    --task t2av \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/metax/minimax_h3_t2av_tp1_block_offload.json" \
    --prompt "${prompt}" \
    --save_result_path "${save_result_path}" \
    --seed 0 \
    --warmup
