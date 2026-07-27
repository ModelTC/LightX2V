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
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i_tp_sp_cfg_flashinfer.json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i_tp2_sp2_cfg2_flashinfer.png" \
    --seed 42
