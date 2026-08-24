#!/bin/bash

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="$(cd -- "${script_dir}/../.." && pwd)"
workspace_path="$(dirname -- "${lightx2v_path}")"
model_path="${HUNYUAN_IMAGE3_MODEL_PATH:-${workspace_path}/HunyuanImage-3-Instruct}"
hunyuan_image3_path="${HUNYUAN_IMAGE3_SOURCE_PATH:-${workspace_path}/HunyuanImage-3.0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${hunyuan_image3_path}:${PYTHONPATH:-}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0

source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL=0
export PYTORCH_ALLOC_CONF="backend:native,expandable_segments:False"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"

torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task ti2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_ti2i_ar_tp4_cudagraph_denoise_fa3_tp2_sp2_multi_micro_flashinfer.json" \
    --prompt "新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。鱼眼镜头，背景是房间门口，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清绒毛细节，面部特写，宝丽莱相纸，写实胶片摄影，复古颗粒感。" \
    --image_path "${model_path}/assets/demo_instruct_imgs/input_0_0.png" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_ti2i_ar_tp4_cudagraph_denoise_fa3_tp2_sp2_multi_micro_flashinfer.png" \
    --seed 42
