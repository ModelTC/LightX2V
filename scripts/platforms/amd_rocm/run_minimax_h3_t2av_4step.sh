#!/bin/bash

# MiniMax-H3 T2AV with the 4-step 768p Turbo LoRA on one AMD Instinct GPU (MI300X / MI350X / MI355X).
# System management interface: amd-smi
#
# Requirements: the ROCm build of PyTorch plus aiter (see lightx2v_platform/base/amd_rocm.py), and the
# Diffusers component layout under model_path (transformer/, text_encoder/, tokenizer/, processor/, vae/,
# audio_vae/). Set the local LoRA path in the JSON config before running.
# The config keeps every component resident: a 288 GB MI355X holds the BF16 DiT, the text encoder and both
# VAEs with room to spare, so no CPU offload or AdaLN cache is needed.

# set path firstly
lightx2v_path=
model_path=

export PLATFORM=amd_rocm
export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

python -m lightx2v.infer \
--model_cls minimax_h3 \
--model-variant fl2av \
--task t2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_t2av_4step.json \
--prompt 'integrated_multimodal_description: A cinematic fox walks through a snowy pine forest at dawn. overall_soundscape: Soft wind, crunching snow, and distant birds. non_diegetic_music: Quiet warm strings.' \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_amd_rocm.mp4 \
--seed 42
