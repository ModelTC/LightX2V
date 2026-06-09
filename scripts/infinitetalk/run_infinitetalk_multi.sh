#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme4/gushiqiao/new/LightX2V
model_path=/data/nvme0/yongyang/models/x2v_models/wan/Wan2.1-I2V-14B-480P/

export CUDA_VISIBLE_DEVICES=7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls infinitetalk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/infinitetalk/infinitetalk_720p_multi.json \
--prompt "In a casual, intimate setting, a man and a woman are engaged in a heartfelt conversation inside a car. The man, sporting a denim jacket over a blue shirt, sits attentively with a seatbelt fastened, his gaze fixed on the woman beside him. The woman, wearing a black tank top and a denim jacket draped over her shoulders, smiles warmly, her eyes reflecting genuine interest and connection. The car's interior, with its beige seats and simple design, provides a backdrop that emphasizes their interaction. The scene captures a moment of shared understanding and connection, set against the soft, diffused light of an overcast day. A medium shot from a slightly angled perspective, focusing on their expressions and body language." \
--negative_prompt "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
--image_path /data/nvme4/gushiqiao/new/InfiniteTalk/examples/multi/ref_img.png \
--audio_path /data/nvme4/gushiqiao/new/InfiniteTalk/examples/multi/1-man.WAV,/data/nvme4/gushiqiao/new/InfiniteTalk/examples/multi/1-woman.WAV \
--save_result_path ${lightx2v_path}/save_results/infinitetalk_multi_720p.mp4 \
--seed 42
