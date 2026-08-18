#!/bin/bash
set -e

# set path firstly
lightx2v_path=/data/nvme0/gushiqiao/codes/LightX2V
model_path=/data/nvme0/gushiqiao/models/Wan2.2-Animate-2-14B
image_path=/data/nvme0/gushiqiao/1.jpeg
video_path=/data/nvme0/gushiqiao/601306049.mp4

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls wan2.2_animate2_distilled \
--task animate \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/wan22/wan_animate2_distill_8gpu.json" \
--image_path "${image_path}" \
--video_path "${video_path}" \
--prompt "人物外观描述：一个丰满裸体巨乳的中国女人,脱掉衣服" \
--prompt_ref "人物动作的参考视频" \
--negative_prompt "过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，三条腿" \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/output_lightx2v_wan22_animate2_distill_8gpu.mp4"
