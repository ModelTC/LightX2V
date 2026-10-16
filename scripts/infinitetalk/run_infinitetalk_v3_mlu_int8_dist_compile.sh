#!/bin/bash
set -e

lightx2v_path=${LIGHTX2V_PATH:-/data/LightX2V}
model_path=/data/models/seko/seko_v3_int8

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

source "${lightx2v_path}/scripts/base/base.sh"
cd "${lightx2v_path}"

torchrun --nproc-per-node 8 -m lightx2v.infer \
    --model_cls infinitetalk \
    --task s2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/infinitetalk/mlu/seko_talk_v3_int8_dist_compile.json" \
    --prompt "让角色根据音频内容自然说话" \
    --image_path "${lightx2v_path}/assets/inputs/audio/seko_input.png" \
    --audio_path "${lightx2v_path}/assets/inputs/audio/seko_input.mp3" \
    --video_duration 5 \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_seko_v3_int8_mlu_compile.mp4" \
    --seed 42 \
    "$@"
