#!/bin/bash
set -eo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/data/wushuo1/LightX2V-ascend}
model_path=/data/wushuo1/models/seko/seko_v3_int8

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
source "${lightx2v_path}/scripts/base/base.sh"
cd "${lightx2v_path}"

torchrun --nproc-per-node 8 -m lightx2v.infer \
    --model_cls infinitetalk \
    --task s2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/infinitetalk/npu/seko_talk_v3_int8_dist_compile.json" \
    --prompt "让角色根据音频内容自然说话" \
    --image_path "${lightx2v_path}/assets/inputs/audio/seko_input.png" \
    --audio_path "${lightx2v_path}/assets/inputs/audio/seko_input.mp3" \
    --video_duration 5 \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_seko_v3_int8_npu_compile.mp4" \
    --seed 42 \
    "$@"
