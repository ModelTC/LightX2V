#!/bin/bash

lightx2v_path=/data/nvme4/gushiqiao/new/LightX2V
model_path=/data/nvme5/gushiqiao/models/SekoTalk-Distill-AR/

export CUDA_VISIBLE_DEVICES=0,2,4,7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc-per-node 4 -m lightx2v.infer \
--model_cls seko_talk_ar \
--task rs2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/ar/seko_talk_ar_kv_kiviquant.json \
--prompt  "In a high-fidelity realistic lifestyle aesthetic, a young woman is captured lounging comfortably on a plush beige sectional sofa within a bright, minimalist interior defined by clean white walls and soft, diffused natural lighting. The subject has shoulder-length chestnut brown hair, striking blue eyes, and is dressed in a cozy, loose-fitting beige knit sweater paired with black pants. She wears a single white wireless earbud in her right ear and a ring on her left hand. Beside her on the sofa cushion lies a closed dark-colored laptop or tablet and a white earbud charging case. Throughout the scene, she leans back in a relaxed posture, her left elbow resting on the top of the sofa cushion with her hand gently supporting her head, while her right hand rests on her drawn-up knee. She is actively speaking, displaying a natural and engaging expression with rhythmic lip movements and subtle facial animations that suggest a casual conversation or vlog recording. Her movements are fluid and grounded; she occasionally shifts her weight slightly against the cushions and uses small, nuanced hand gestures with her right hand, lifting it briefly from her knee to emphasize her words before settling it back down. The camera maintains a fixed, static medium shot, framing her centrally to capture her upper body and the immediate cozy environment, creating an intimate and serene atmosphere without any camera movement or shifts in focus." \
--negative_prompt "low quality,blurry,pixelated,low resolution,noise,artifacts,poor lighting, overexposed, underexposed, distorted, unnatural, deformed, weird,scared,anatomy,mutated, wrong proportions, extra limbs,floating objects, disconnected, gravity-defying, impossible shadows,wrong lighting,non-existent reflections,inconsistent perspective, repetitive, monotonous, monotonous, generic, watermark, ugly, high contrast, bad photo, font, username, error, logo, words, letters, digits, autograph, trademark, name, twisted face, (poorly drawn hands, malformed hands, missing fingers, unnatural hand positions, blur hand, multiple fingers, multiple arms), static, naked, artifacts, oversaturated" \
--image_path "/data/nvme4/gushiqiao/new/example/1_素材图.png" \
--audio_path "/data/nvme4/gushiqiao/new/example/1_素材图.mp3" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk_ar_sp2.mp4 \
--seed 0
