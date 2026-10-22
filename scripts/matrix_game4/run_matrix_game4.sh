#!/bin/bash

lightx2v_path=/data/nvme6/gushiqiao/codes/report/LightX2V
model_path=/data/nvme3/gushiqiao/models/Matrix-Game-4.0
: "${VIDEO_PATH:?Set VIDEO_PATH to the source game recording}"
video_path="${VIDEO_PATH}"
image_path="${model_path}/models/castle/sink.png"
config_path="${CONFIG_PATH:-${lightx2v_path}/configs/matrix_game4/quality.json}"
num_gpus="${NPROC_PER_NODE:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PROFILING_DEBUG_LEVEL=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${lightx2v_path}"

python -m lightx2v.infer \
  --model_cls matrix_game4 \
  --model-variant castle \
  --task v2v \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --video_path "${video_path}" \
  --image_path "${image_path}" \
  --prompt 'a third-person shooter game in a medieval castle with stone walls, towers and courtyards, stylized cartoon characters firing glowing projectiles, trees and grass, snowy mountains in the distance, realistic AAA game rendering, physically based materials, natural daylight, detailed vegetation and terrain' \
  --seed 42 \
  --save_result_path "${lightx2v_path}/save_results/matrix_game4_castle.mp4"
