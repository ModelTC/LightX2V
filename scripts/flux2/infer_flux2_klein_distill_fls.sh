#!/bin/bash
lightx2v_path=
model_path="/data/temp/black-forest-labs/FLUX.2-klein-9B"
export CUDA_VISIBLE_DEVICES=3

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls flux2_klein \
    --task t2i \
    --target_shape 1024 1024 \
    --model_path $model_path \
    --prompt "A graceful Chinese woman wearing an elegant embroidered qipao with delicate woven textures and subtle solid-color silk fabric, standing beneath a large ancient tree at sunset. Warm golden sunlight filters through the leaves and gently falls across her qipao, creating soft highlights and rich shadows. Traditional oriental aesthetics, cinematic lighting, ultra detailed fabric texture, calm and poetic atmosphere, natural pose, soft breeze moving her hair, highly realistic, masterpiece, shallow depth of field, golden hour photography, serene and timeless mood." \
    --save_result_path "${lightx2v_path}/save_results/flux2_klein_distill_fls.png" \
    --config_json "${lightx2v_path}/configs/flux2/flux2_klein_distill_fls.json"
