#!/bin/bash
lightx2v_path=

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None
export PROFILING_DEBUG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0

python -m lightx2v.shot_runner.stream_infer \
--config_json ${lightx2v_path}/configs/seko_talk/shot/stream/main.json \
--prompt  "The video features a male speaking to the camera with arms spread out, a slightly furrowed brow, and a focused gaze." \
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_seko_talk_shot_stream.mp4
