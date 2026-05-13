#!/bin/bash
#
# LTX-2.3 v2av motion transfer — Union-Control IC-LoRA.
# Accepts any of: Canny / Depth / Pose mode

# set path firstly
lightx2v_path=
model_path=
video_path=
image_path=

export CUDA_VISIBLE_DEVICES=0

# DWPose ONNX checkpoint root. Expected layout:
#   ${dwpose_ckpt_path}/det/yolox_l.onnx
#   ${dwpose_ckpt_path}/pose2d/dw-ll_ucoco_384.onnx
# (Download once from https://huggingface.co/yzd-v/DWPose)
dwpose_ckpt_path=/path/to/dwpose
process_out_dir=${lightx2v_path}/save_results/ltx2_v2av/process_results

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# 1) Preprocess driving video into a pose-skeleton mp4 (DWPose).
python ${lightx2v_path}/tools/preprocess/ltx2.3_preprocess_data.py \
    --ckpt_path ${dwpose_ckpt_path} \
    --video_path ${video_path} \
    --refer_path ${image_path} \
    --save_path ${process_out_dir}/pose_skeleton.mp4 \
    --resolution_area 1280 768 \
    --fps -1 \
    --bg_mode black \
    --include_hands \
    --include_face \
    --mode pose \
    --device cuda

# 2) Run LTX-2.3 v2av.
python -m lightx2v.infer \
--model_cls ltx2 \
--task v2av \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_3_v2av_motion_union.json \
--image_path "${image_path}" \
--video_path "${process_out_dir}/pose_skeleton.mp4" \
--mux_audio_video_path "${video_path}" \
--reference_video_strength 1 \
--prompt "角色自然地做动作和表情" \
--negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts." \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_ltx2_3_v2av_motion_pose.mp4 \
--image_strength 1.0 \
