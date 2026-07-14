#!/bin/bash
# Full-quality S2V; aligned with Wan2.2/run_s2v_full.sh (full audio, 40 steps).

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.2-S2V-14B

export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.2_s2v \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan22/wan_s2v.json \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
--negative_prompt "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--image_path Wan2.2/examples/i2v_input.JPG \
--audio_path Wan2.2/examples/talk.wav \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_s2v_full.mp4
