#!/bin/bash

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme5/gushiqiao/codes/LightX2V}
model_path=${MODEL_PATH:-/data/nvme5/gushiqiao/models/Cosmos3-Super-Text2Image}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls cosmos3 \
--task t2i \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/cosmos3/cosmos3_super_t2i.json \
--prompt "一只小猫" \
--negative_prompt "${NEGATIVE_PROMPT:-}" \
--save_result_path "${SAVE_PATH:-${lightx2v_path}/save_results/cosmos3_t2i.png}" \
--seed ${SEED:-1143}
