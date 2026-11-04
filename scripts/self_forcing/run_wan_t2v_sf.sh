#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/../../sf/wan_models/Wan2.1-T2V-1.3B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls wan2.1_sf \
--task t2v \
--model_path "${model_path}" \
--config_json "${CONFIG_PATH:-${lightx2v_path}/configs/self_forcing/wan_t2v_sf_bf16_lighttae_5090.json}" \
--prompt 'A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides.' \
--seed "${SEED:-0}" \
--save_result_path "${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_wan_t2v_sf.mp4}"
