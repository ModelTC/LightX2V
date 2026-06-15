#!/bin/bash

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme4/gushiqiao/new/LightX2V}
model_path=${MODEL_PATH:-/data/nvme5/gushiqiao/models/ERNIE-Image}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls ernie_image \
--task t2i \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/ernie_image/ernie_image_t2i.json \
--prompt "${PROMPT:-一只黑白相间的中华田园犬}" \
--negative_prompt "${NEGATIVE_PROMPT:-}" \
--save_result_path "${SAVE_PATH:-${lightx2v_path}/save_results/ernie_image_t2i.png}" \
--seed ${SEED:-42}
