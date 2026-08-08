#!/bin/bash
set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
model_path=${MODEL_PATH:-}
image_path=${IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/img_0.jpg}
save_result_path=${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/wan_i2v_sol_attn.mp4}

if [[ -z "${model_path}" ]]; then
    echo "MODEL_PATH must point to a Wan2.1 I2V checkpoint." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SOL_ATTN_STRICT=1

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls wan2.1 \
    --task i2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/attentions/wan_i2v_sol_attn.json" \
    --prompt "${PROMPT:-A fluffy orange cat sits by a window, cinematic lighting, detailed fur, gentle camera movement.}" \
    --negative_prompt "${NEGATIVE_PROMPT:-Low quality, blurry, distorted anatomy, duplicate limbs, static frame, watermark, text.}" \
    --image_path "${image_path}" \
    --save_result_path "${save_result_path}"
