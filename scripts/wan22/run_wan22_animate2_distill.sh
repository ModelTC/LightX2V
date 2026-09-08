#!/bin/bash
set -e

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.2-Animate-2-14B
image_path=/path/to/Wan-Animate-2/examples/demo1/reference.png
video_path=/path/to/Wan-Animate-2/examples/demo1/template.mp4

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls wan2.2_animate2_distilled \
--task animate \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/wan22/wan_animate2_distill.json" \
--image_path "${image_path}" \
--video_path "${video_path}" \
--prompt "人物外观描述：一只银灰色虎斑纹的小猫，拥有圆润的脸庞、竖立的耳朵和巨大的圆形眼睛。它身穿一套深蓝色的制服套装，包括一件带有金色纽扣的西装外套和一条百褶裙。外套里面搭配着白色衬衫，领口处系着一个红色的蝴蝶结，袖口露出白色的衬衫边缘。背景描述：背景为纯白色，光线均匀明亮，无其他杂物或装饰。" \
--prompt_ref "人物动作的参考视频" \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/output_lightx2v_wan22_animate2_distill.mp4"
