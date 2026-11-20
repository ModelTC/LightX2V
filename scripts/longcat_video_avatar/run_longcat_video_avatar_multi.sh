#!/bin/bash
set -eo pipefail

lightx2v_path=/path/to/LightX2V
model_path=/path/to/LongCat-Video-Avatar-1.6

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16

python -m lightx2v.infer \
  --model_cls longcat_video_avatar \
  --task s2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/longcat_video_avatar/h800/s2v_bf16.json" \
  --prompt "Two people talking to each other." \
  --image_path assets/inputs/audio/multi_person/seko_input.png \
  --audio_path '["assets/inputs/audio/multi_person/p1.mp3", "assets/inputs/audio/multi_person/p2.mp3"]' \
  --save_result_path "${lightx2v_path}/save_results/longcat_video_avatar_multi.mp4"
