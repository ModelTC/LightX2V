#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${repo_root}/../../sf/wan_models/Wan2.1-T2V-14B}"
config_path="${CONFIG_PATH:-${repo_root}/configs/realtime_video/realtime_fp8_f16_accum_sage_lighttae_5090_sp8.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

cd "${repo_root}"
torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls wan2.1_realtime \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --prompt "${PROMPT:-A hyperrealistic close-up of ocean waves shimmering at sunset.}" \
  --seed "${SEED:-42}" \
  --save_result_path "${OUTPUT_PATH:-${repo_root}/save_results/krea_realtime_5090.mp4}"
