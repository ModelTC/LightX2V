#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/../models/Matrix-Game-4.0}"
: "${VIDEO_PATH:?Set VIDEO_PATH to the source game recording}"
video_path="${VIDEO_PATH}"
image_path="${IMAGE_PATH:-${model_path}/models/castle/sink.png}"
config_path="${CONFIG_PATH:-${lightx2v_path}/configs/matrix_game4/realtime_fp8_f16_accum_sage_lighttae_5090.json}"
num_gpus="${NPROC_PER_NODE:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:$PYTHONPATH}"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

cd "${lightx2v_path}"

launcher=(python)
if [[ "$num_gpus" -gt 1 ]]; then
  launcher=(torchrun --nproc_per_node "$num_gpus")
fi
"${launcher[@]}" -m lightx2v.infer \
  --model_cls matrix_game4 \
  --model-variant castle \
  --task v2v \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --video_path "${video_path}" \
  --image_path "${image_path}" \
  --prompt 'a third-person shooter game in a medieval castle with stone walls, towers and courtyards, stylized cartoon characters firing glowing projectiles, trees and grass, snowy mountains in the distance, realistic AAA game rendering, physically based materials, natural daylight, detailed vegetation and terrain' \
  --seed "${SEED:-0}" \
  --save_result_path "${OUTPUT_PATH:-${lightx2v_path}/save_results/matrix_game4_castle.mp4}"
