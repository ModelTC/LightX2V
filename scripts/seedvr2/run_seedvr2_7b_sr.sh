#!/bin/bash

# set path and first
lightx2v_path=/path/to/LightX2V

# Use the 3B model repo because the official 7B repo is missing pos_emb.pt and neg_emb.pt (e.g. https://huggingface.co/ByteDance-Seed/SeedVR2-3B/blob/main/pos_emb.pt); specify the 7B DiT checkpoint via dit_original_ckpt in configs/seedvr/seedvr2_7b.json.
model_path=/path/to/ByteDance-Seed/SeedVR2-3B

video_path=/path/to/test.mp4

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls seedvr2 \
--task sr \
--sr_ratio 2.0 \
--video_path $video_path \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seedvr/seedvr2_7b.json \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seedvr2_7b_sr.mp4
