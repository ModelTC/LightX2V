#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
model_path="${model_path:-/data/nvme0/lhd_codes/SenseNova-Vision/models/SenseNova-Vision-7B-MoT}"
SENSENOVA_SOURCE_PATH="${SENSENOVA_SOURCE_PATH:-/data/nvme0/lhd_codes/SenseNova-Vision}"
TASK="${1:-depth}"

export lightx2v_path model_path
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
SENSENOVA_MODE="${SENSENOVA_MODE:-}"

if [[ "${TASK}" == "understanding" ]]; then
    TASK="raw_query"
    SENSENOVA_MODE="understanding"
    PROMPT="${PROMPT:-What are the main objects in this scene and their relationships?}"
    SAVE_PATH="${SAVE_PATH%.*}.txt"
fi

mkdir -p "$(dirname "${SAVE_PATH}")"
extra_args=()
if [[ -n "${RAW_OUTPUT_PATH:-}" ]]; then
    extra_args+=(--raw_output_path "${RAW_OUTPUT_PATH}")
fi
if [[ -n "${GLB_OUTPUT_PATH:-}" ]]; then
    extra_args+=(--glb_output_path "${GLB_OUTPUT_PATH}")
fi
if [[ "${POSTPROCESS_PREDICTIONS:-false}" == "true" ]]; then
    extra_args+=(--postprocess_predictions)
fi

python -m lightx2v.infer \
    --model_cls sensenova_vision \
    --task "${TASK}" \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/sensenova_vision/sensenova_vision.json" \
    --image_path "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --sensenova_mode "${SENSENOVA_MODE}" \
    --sensenova_source_path "${SENSENOVA_SOURCE_PATH}" \
    --save_result_path "${SAVE_PATH}" \
    --seed "${SEED:-42}" \
    "${extra_args[@]}"
