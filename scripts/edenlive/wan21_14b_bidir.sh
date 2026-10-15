#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/EdenLive
dit_original_ckpt=/path/to/EdenLive-x2v/dit-14b-bidir
wan_path=/path/to/Wan2.1-I2V-14B-720P
recording_path=/path/to/recording

export CUDA_VISIBLE_DEVICES=0
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
  --model_cls edenlive \
  --task i2v \
  --model_path "$model_path" \
  --dit_original_ckpt "$dit_original_ckpt" \
  --vae_path "$model_path/vae-14b/Wan2.1_VAE.pth" \
  --t5_original_ckpt "$wan_path/models_t5_umt5-xxl-enc-bf16.pth" \
  --clip_original_ckpt "$wan_path/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --t5_tokenizer_path "$model_path/umt5/tokenizer" \
  --config_json "${lightx2v_path}/configs/edenlive/wan21_14b_bidir.json" \
  --prompt "$recording_path/prompts.json" \
  --negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  --image_path "$recording_path/cover.png" \
  --video_path "$recording_path/recording.mp4" \
  --seed 42 \
  --save_result_path "${lightx2v_path}/save_results/edenlive_14b_bidir.mp4"
