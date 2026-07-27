#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../../../.." && pwd)}"
model_path="${MODEL_PATH:-/data/wushuo1/models/Wan2.2-I2V-A14B}"
# CANN 8.5.1 lacks aclnnBlockSparseAttention, so this config intentionally
# keeps the same dynamic Top-K mask algorithm on the supported MindIE RF v2 op.
config_path="${lightx2v_path}/configs/platforms/ascend_npu/extreme/wan_moe_i2v_distill_int8_dynamic_sparse_attn.json"

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}"

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls wan2.2_moe_distill \
    --task i2v \
    --warmup \
    --model_path "${model_path}" \
    --config_json "${config_path}" \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_wan22_moe_i2v_int8_dynamic_sparse_npu.mp4"
