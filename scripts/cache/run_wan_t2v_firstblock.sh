#!/bin/bash

# set path and first
lightx2v_path="/home/yangrongjin/lightx2v"
model_path="/mtc/yongyang/models/x2v_models/wan/Wan2.1-T2V-1.3B"

# check section
export CUDA_VISIBLE_DEVICES=2
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export DTYPE=BF16
export ENABLE_PROFILING_DEBUG=true
export ENABLE_GRAPH_MODE=false

python -m lightx2v.infer \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/caching/firstblock/wan_t2v_1_3b.json \
--prompt "A cinematic wide-angle shot of a professional skier in a redand white suit racing down a snow-covered mountain slope at high speed, snowflying around, sunlight reflecting off the icy surface, with pine treesblurring in the background and a clear blue sky above." \
--negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--save_video_path ${lightx2v_path}/save_results/firstblock_t2v.mp4
