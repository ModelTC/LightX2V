#!/bin/bash
lightx2v_path=

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None
export PROFILING_DEBUG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc-per-node 2 -m lightx2v.shot_runner.rs2v_infer \
--config_json ${lightx2v_path}/configs/seko_talk/shot/rs2v/main_dist.json \
--prompt  "A cultured woman speaking passionately and eloquently, her expression alive with emotion, conveying resolve, dignity, and a deep sense of purpose." \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_seko_talk_shot_rs2v.mp4
