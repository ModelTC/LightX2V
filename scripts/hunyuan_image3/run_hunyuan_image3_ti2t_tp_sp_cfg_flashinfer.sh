#!/bin/bash

# Set paths first.
lightx2v_path=/data/nvme0/lhd_codes/LightX2V
model_path=/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct
hunyuan_image3_path=/data/nvme0/lhd_codes/HunyuanImage-3.0

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${hunyuan_image3_path}:${PYTHONPATH:-}"

source "${lightx2v_path}/scripts/base/base.sh"

torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task ti2t \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_ti2t_tp_sp_cfg_flashinfer.json" \
    --prompt "请描述图像中的主要内容和视觉风格。" \
    --image_path "${hunyuan_image3_path}/assets/demo_instruct_imgs/input_0_0.png" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_ti2t_tp2_sp2_cfg1_flashinfer.txt" \
    --seed 42
