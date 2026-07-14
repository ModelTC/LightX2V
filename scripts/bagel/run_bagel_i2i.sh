#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/BAGEL-7B-MoT}"
IMAGE_PATH="${IMAGE_PATH:-assets/inputs/imgs/img_0.jpg}"
PROMPT="${PROMPT:-Change the scene to golden hour while preserving the main subject.}"
SAVE_PATH="${SAVE_PATH:-save_results/bagel_i2i.png}"

python -m lightx2v.infer \
  --model_cls bagel \
  --task i2i \
  --model_path "${MODEL_PATH}" \
  --config_json configs/bagel/bagel_i2i.json \
  --image_path "${IMAGE_PATH}" \
  --prompt "${PROMPT}" \
  --seed 42 \
  --save_result_path "${SAVE_PATH}"
