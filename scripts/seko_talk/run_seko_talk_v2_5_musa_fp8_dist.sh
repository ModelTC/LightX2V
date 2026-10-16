#!/bin/bash
set -e

lightx2v_path=/data/wushuo_musa/LightX2V-musa
model_path=/data/models/seko/seko_v2.5_fp8

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32

source "${lightx2v_path}/scripts/base/base.sh"
cd "${lightx2v_path}"

torchrun --nproc-per-node 8 -m lightx2v.infer \
    --model_cls seko_talk \
    --task s2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/seko_talk/musa/seko_talk_v2_5_fp8_dist.json" \
    --prompt "让角色根据音频内容自然说话" \
    --image_path "${lightx2v_path}/assets/inputs/audio/seko_input.png" \
    --audio_path "${lightx2v_path}/assets/inputs/audio/seko_input.mp3" \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_seko_v2.5_fp8_musa.mp4" \
    --seed 42 \
    "$@"
