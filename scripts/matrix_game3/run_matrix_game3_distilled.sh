#!/bin/bash
# Run Matrix-Game-3.0 distilled model inference via LightX2V
# Usage: ./run_matrix_game3_distilled.sh

# Set model path (update this to your local Matrix-Game-3.0 model directory)
MODEL_PATH="${MODEL_PATH:-/path/to/Matrix-Game-3.0}"
CONFIG_JSON="configs/matrix_game3/matrix_game3_distilled.json"
SAVE_PATH="${SAVE_PATH:-save_results/matrix_game3_distilled}"

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
