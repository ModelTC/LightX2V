#!/bin/bash

lightx2v_path=
model_path=

export PLATFORM=mps
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls minimax_h3 \
    --model-variant fl2av \
    --task t2av \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/mps/minimax_h3_t2av_4step_512_22.json" \
    --prompt "A cinematic fox walking through a snowy forest" \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av.mp4" \
    --seed 42
