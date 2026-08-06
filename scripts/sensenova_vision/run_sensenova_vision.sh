#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
model_path="${model_path:-/data/nvme0/lhd_codes/SenseNova-Vision/models/SenseNova-Vision-7B-MoT}"
SENSENOVA_SOURCE_PATH="${SENSENOVA_SOURCE_PATH:-/data/nvme0/lhd_codes/sensenova-vision-v2}"
TASK="${1:-depth}"

export lightx2v_path model_path SENSENOVA_SOURCE_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"

if [[ "${TASK}" == "example" ]]; then
    exec python "${lightx2v_path}/examples/sensenova_vision/example_visualize.py" \
        --model_path "${model_path}" \
        --source_path "${SENSENOVA_SOURCE_PATH}" \
        --output_dir "${OUTPUT_DIR:-${lightx2v_path}/save_results/sensenova_vision_example}" \
        --seed "${SEED:-42}" \
        --example "${EXAMPLE_ID:-all}"
fi

IMAGE_PATH="${IMAGE_PATH:-${SENSENOVA_SOURCE_PATH}/examples/images/2.jpg}"
PROMPT="${PROMPT:-}"
SAVE_PATH="${SAVE_PATH:-${lightx2v_path}/save_results/sensenova_${TASK}.png}"

if [[ "${TASK}" == "understanding" ]]; then
    PROMPT="${PROMPT:-What are the main objects in this scene and their relationships?}"
    SAVE_PATH="${SAVE_PATH%.*}.txt"
fi

mkdir -p "$(dirname "${SAVE_PATH}")"

python -m lightx2v.infer \
    --model_cls sensenova_vision \
    --task omni_vision_task \
    --omni_vision_subtask "${TASK}" \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/sensenova_vision/sensenova_vision.json" \
    --image_path "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --save_result_path "${SAVE_PATH}" \
    --seed "${SEED:-42}"
