#!/usr/bin/env bash
set -euo pipefail

lightx2v_path=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

# Replace these placeholder paths, or override them through environment variables.
# Base model: https://huggingface.co/MiniMaxAI/MiniMax-H3
# Use the Diffusers component layout: transformer/, text_encoder/, tokenizer/, processor/, vae/, audio_vae/.
model_path=${MODEL_PATH:-/path/to/MiniMax-H3}
image_path=${IMAGE_PATH:-/path/to/input.png}
# In the config JSON below, set lora_configs[0].path to your local step-10000.safetensors.
# H3-World LoRA: https://huggingface.co/DANNY621/H3-World/blob/main/step-10000.safetensors
config_path=${CONFIG_JSON:-${lightx2v_path}/configs/minimax_h3_world/minimax_h3_world.json}
prompt=${PROMPT:-A third-person view of a man in a light gray shirt and jeans beside a low concrete roadside barrier, with a tall concrete retaining wall and vegetation behind it in daylight.}
seed=${SEED:-2}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=${PYTHONPATH:-}
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls minimax_h3_world \
  --model-variant fl2av \
  --task ia2av \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --image_path "${image_path}" \
  --prompt "${prompt}" \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_minimax_h3_world.mp4" \
  --seed "${seed}"
