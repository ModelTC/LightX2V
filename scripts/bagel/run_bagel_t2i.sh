#!/bin/bash

# set path and first
export lightx2v_path=/path/to/LightX2V
export model_path=/path/to/ByteDance-Seed/BAGEL-7B-MoT

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls bagel \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/bagel/bagel_t2i.json \
    --prompt "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere." \
    --negative_prompt " " \
    --save_result_path ${lightx2v_path}/save_results/bagel_t2i.png \
    --seed 42
