#!/bin/bash

# set path and first
lightx2v_path=/path/to/lightx2v
model_path=/path/to/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v
# model_path=/path/to/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v

# check section
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

export PROFILING_DEBUG_LEVEL=2
export ENABLE_GRAPH_MODE=false
export DTYPE=BF16

python -m lightx2v.infer \
--model_cls wan2.1_distill \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/bench/lightx2v_6_distill.json \
--prompt "A close-up cinematic view of a person cooking in a warm,sunlit kitchen, using a wooden spatula to stir-fry a colorful mix of freshvegetables—carrots, broccoli, and bell peppers—in a black frying pan on amodern induction stove. The scene captures the glistening texture of thevegetables, steam gently rising, and subtle reflections on the stove surface.In the background, soft-focus jars, fruits, and a window with natural daylightcreate a cozy atmosphere. The hand motions are smooth and rhythmic, with a realisticsense of motion blur and lighting." \
--negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_2.jpg \
--save_video_path ${lightx2v_path}/save_results/lightx2v_6_distill.mp4
