#!/bin/bash
# Pose + audio S2V, 8-GPU Ulysses (seq_p_size=8).

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.2-S2V-14B

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls wan2.2_s2v \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/dist_infer/wan22_s2v_pose_audio_dist_ulysses.json \
--seed 42 \
--prompt "a person is singing" \
--negative_prompt "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--image_path Wan2.2/examples/pose.png \
--audio_path Wan2.2/examples/sing.MP3 \
--src_pose_path Wan2.2/examples/pose.mp4 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_s2v_pose_audio_dist.mp4
