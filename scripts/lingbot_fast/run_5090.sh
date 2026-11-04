#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${repo_root}/../models/lingbot-world-v2-14b-causal-fast}"
example_root="${EXAMPLE_ROOT:-${repo_root}/../lingbot-world-v2-source-20261025/examples/03}"
config_path="${CONFIG_PATH:-${repo_root}/configs/lingbot_fast/realtime_v2_fp8_f16_accum_sage_lighttae_5090_sp8.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

cd "${repo_root}"
torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls lingbot_world_fast \
  --task i2v \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --prompt "${PROMPT:-A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere.}" \
  --image_path "${IMAGE_PATH:-${example_root}/image.jpg}" \
  --action_path "${ACTION_PATH:-${example_root}}" \
  --seed "${SEED:-42}" \
  --save_result_path "${OUTPUT_PATH:-${repo_root}/save_results/lingbot_v2_5090.mp4}"
