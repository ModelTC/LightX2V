#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/InfiniteTalk

# Set adapter_model_path and audio_encoder_path in the JSON before starting.
export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.server \
  --model_cls infinitetalk \
  --task s2v \
  --model_path $model_path \
  --config_json ${lightx2v_path}/configs/infinitetalk/infinitetalk_480p_multi_reuse.json \
  --host 0.0.0.0 \
  --port 8000
