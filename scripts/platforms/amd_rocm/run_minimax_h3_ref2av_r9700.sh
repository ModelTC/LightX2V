#!/bin/bash

# MiniMax-H3 Ref2AV on 2x AMD Radeon AI PRO R9700 (gfx1201), Ulysses sequence parallel.
# Attention uses the hip_sage_amd_rocm backend: INT4 Q.K^T for DiT blocks 0-44 and INT8 for blocks 45-49
# (hip_sage_setting in the config). The HIP kernels are built with hipcc on first use (HIP >= 7.15 recommended)
# and cached under ~/.cache/lightx2v/hip_sage.
# System management interface: amd-smi

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export PLATFORM=amd_rocm
export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
    --model_cls minimax_h3 \
    --model-variant ref2av \
    --task ref2av \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_ref2av_sp2_r9700_hip_sage.json" \
    --prompt "Generate an audio-video scene following the references." \
    --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
    --save_result_path "${lightx2v_path}/save_results/minimax_h3_ref2av_r9700_hip_sage.mp4" \
    --seed 42
