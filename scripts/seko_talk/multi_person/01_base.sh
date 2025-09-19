#!/bin/bash

lightx2v_path=/mnt/afs2/wangshankun/lightx2v
model_path=/mnt/afs1/gushiqiao/Wan2.1-R2V0909-Audio-14B-720P

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SENSITIVE_LAYER_DTYPE=None

python  -m debugpy --listen 0.0.0.0:15678  -m lightx2v.infer \
--model_cls seko_talk \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/mutil_person/01_base.json \
--prompt  "The video features a man and a woman standing by a bench in the park, their expressions tense and voices raised as they argue. The man gestures with both hands, his arms swinging slightly as if to emphasize each heated word, while the woman stands with her hands on her waist, her brows furrowed in frustration. The background is a wide expanse of sunlit grass, the golden light contrasting with the sharp energy of their quarrel. Their voices seem to clash in the air, and the rhythm of their hand movements and body postures interweaves with the rising tension, creating a vivid scene of confrontation." \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path ${lightx2v_path}/assets/inputs/audio/multi_person/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/multi_person \
--person_mask_path ${lightx2v_path}/assets/inputs/audio/multi_person \
--save_video_path ${lightx2v_path}/save_results/seko_talk_multi_person.mp4
