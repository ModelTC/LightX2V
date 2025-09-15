#!/bin/bash

# set path and first
lightx2v_path=
model_path=
export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export SENSITIVE_LAYER_DTYPE=None

python -m lightx2v.infer \
--model_cls wan2.2_moe_distill \
--task flf2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan22/wan_distill_moe_flf2v_fp8.json \
--prompt "A bearded man with red facial hair wearing a yellow straw hat and dark coat in Van Gogh's self-portrait style, slowly and continuously transforms into a space astronaut. The transformation flows like liquid paint - his beard fades away strand by strand, the yellow hat melts and reforms smoothly into a silver space helmet, dark coat gradually lightens and restructures into a white spacesuit. The background swirling brushstrokes slowly organize and clarify into realistic stars and space, with Earth appearing gradually in the distance. Every change happens in seamless waves, maintaining visual continuity throughout the metamorphosis.\n\nConsistent soft lighting throughout, medium close-up maintaining same framing, central composition stays fixed, gentle color temperature shift from warm to cool, gradual contrast increase, smooth style transition from painterly to photorealistic. Static camera with subtle slow zoom, emphasizing the flowing transformation process without abrupt changes." \
--negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--image_path /mtc/gushiqiao/llmc_workspace/wan22_14B_flf2v_start_image.png \
--last_frame_path /mtc/gushiqiao/llmc_workspace/wan22_14B_flf2v_end_image.png \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_flf2v.mp4
