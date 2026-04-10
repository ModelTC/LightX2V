#!/bin/bash
# Run Matrix-Game-3.0 base model inference via LightX2V
# Usage: ./run_matrix_game3_base.sh

# Set model path (update this to your local Matrix-Game-3.0 model directory)
MODEL_PATH="${MODEL_PATH:-/path/to/Matrix-Game-3.0}"
CONFIG_JSON="configs/matrix_game3/matrix_game3_base.json"
SAVE_PATH="${SAVE_PATH:-save_results/matrix_game3_base}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Reuse the repo's standard runtime defaults. MG3 base needs localized fp32 math
# around time modulation and residuals, but broad fp32-sensitive-layer overrides
# skew the main bf16 execution path away from the official implementation.
export lightx2v_path="${lightx2v_path:-${REPO_ROOT}}"
export model_path="${model_path:-${MODEL_PATH}}"
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls wan2.2_matrix_game3 \
    --task i2v \
    --model_path "${MODEL_PATH}" \
    --config_json "${CONFIG_JSON}" \
    --prompt "a city street scene with cars and pedestrians" \
    --image_path "${IMAGE_PATH:-Matrix-Game-3/Matrix-Game-3/demo_images/001/image.png}" \
    --action_path "${ACTION_PATH:-}" \
    --save_result_path "${SAVE_PATH}" \
    --seed 42
