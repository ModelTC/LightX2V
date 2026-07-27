#!/bin/bash

# Set paths first.
lightx2v_path=/data/nvme0/lhd_codes/LightX2V
model_path=/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct
hunyuan_image3_path=/data/nvme0/lhd_codes/HunyuanImage-3.0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${hunyuan_image3_path}:${PYTHONPATH:-}"

source "${lightx2v_path}/scripts/base/base.sh"

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task ti2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_ti2i_tp_sp_cfg_flashinfer.json" \
    --prompt "新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。鱼眼镜头，背景是房间门口，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清绒毛细节，面部特写，宝丽莱相纸，写实胶片摄影，复古颗粒感。" \
    --image_path "${hunyuan_image3_path}/assets/demo_instruct_imgs/input_0_0.png" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_ti2i_tp2_sp2_cfg2_flashinfer.png" \
    --seed 42
