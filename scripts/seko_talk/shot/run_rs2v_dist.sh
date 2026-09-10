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
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_seko_talk_shot_rs2v.mp4
