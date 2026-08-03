#!/usr/bin/env bash
# set -euo pipefail

export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export model_path="${model_path:-/data/nvme0/lhd_codes/Bagel/models/BAGEL-7B-MoT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROMPT="${PROMPT:-A female cosplayer portraying an ethereal fairy or elf in a magical forest with glowing plants.}"
ASPECT_RATIO="${ASPECT_RATIO:-1:1}"
SAVE_PATH="${SAVE_PATH:-${lightx2v_path}/save_results/bagel_t2i_cot.png}"

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls bagel \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/bagel/bagel_t2i_cot.json" \
    --prompt "${PROMPT}" \
    --aspect_ratio "${ASPECT_RATIO}" \
    --save_result_path "${SAVE_PATH}" \
    --seed "${SEED:-42}"
