#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme4/gushiqiao/new/LightX2V
model_path=/data/nvme0/yongyang/models/x2v_models/wan/Wan2.1-I2V-14B-480P/

export CUDA_VISIBLE_DEVICES=6


# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls infinitetalk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/infinitetalk/infinitetalk_480p_single_distilled.json \
--prompt "A woman is passionately singing into a professional microphone in a recording studio. She wears large black headphones and a dark cardigan over a gray top. Her long, wavy brown hair frames her face as she looks slightly upwards, her mouth open mid-song. The studio is equipped with various audio equipment, including a mixing console and a keyboard, with soundproofing panels on the walls. The lighting is warm and focused on her, creating a professional and intimate atmosphere. A close-up shot captures her expressive performance." \
--negative_prompt "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
--image_path /data/nvme4/gushiqiao/new/InfiniteTalk/examples/single/ref_image.png \
--audio_path /data/nvme4/gushiqiao/new/InfiniteTalk/examples/single/1.wav \
--save_result_path ${lightx2v_path}/save_results/infinitetalk_single_480p.mp4 \
--seed 42
