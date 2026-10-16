#!/bin/bash
set -e

lightx2v_path=/data/wushuo_metax/LightX2V-metax
model_path=/data/models/seko/seko_v3_int8
test_data_path=/data/wushuo_metax/data/test_data/s2v_5s

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

source "${lightx2v_path}/scripts/base/base.sh"

torchrun --standalone --nproc-per-node 8 -m lightx2v.infer \
    --model_cls infinitetalk \
    --task s2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/infinitetalk/metax/seko_talk_v3_int8_dist_compile.json" \
    --prompt "自然说话，表情甜美可爱，头部轻微动作，肢体放松协调，符合日本JK少女气质" \
    --image_path "${test_data_path}/bebe5dab-7204-47a8-808f-1b9a1f2c2940-input_image.png" \
    --audio_path "${test_data_path}/segment-f1571648.wav" \
    --video_duration 5 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_seko_v3_int8_metax_compile.mp4" \
    "$@"
