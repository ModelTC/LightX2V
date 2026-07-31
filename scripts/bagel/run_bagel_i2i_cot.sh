#!/usr/bin/env bash
# set -euo pipefail

export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export model_path="${model_path:-/data/nvme0/lhd_codes/Bagel/models/BAGEL-7B-MoT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IMAGE_PATH="${IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/img_0.jpg}"
PROMPT="${PROMPT:-Change the scene to golden hour while preserving the main subject.}"
SAVE_PATH="${SAVE_PATH:-${lightx2v_path}/save_results/bagel_i2i_cot.png}"

source "${lightx2v_path}/scripts/base/base.sh"
mkdir -p "$(dirname "${SAVE_PATH}")"

python -m lightx2v.infer \
    --model_cls bagel \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/bagel/bagel_i2i_cot.json" \
    --image_path "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --seed "${SEED:-42}" \
    --save_result_path "${SAVE_PATH}"
