#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/MiniMax-H3}
lora_path=${LORA_PATH:-/llm/models/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors}
config_template=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/intel_xpu/minimax_h3_t2av_dmd_lora_4step.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_dmd_lora_4step.mp4}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export PLATFORM=${PLATFORM:-intel_xpu}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH=${PYTHONPATH:-}

[[ -d "${model_path}" ]] || { echo "Model directory not found: ${model_path}" >&2; exit 1; }
[[ -f "${lora_path}" ]] || { echo "LoRA checkpoint not found: ${lora_path}" >&2; exit 1; }
[[ -f "${config_template}" ]] || { echo "Config file not found: ${config_template}" >&2; exit 1; }

mkdir -p "$(dirname -- "${output_path}")"
runtime_config=$(mktemp "${TMPDIR:-/tmp}/lightx2v-minimax-h3-xpu-XXXXXX.json")
trap 'rm -f -- "${runtime_config}"' EXIT

python - "${config_template}" "${runtime_config}" "${lora_path}" <<'PY'
import json
import sys

source, destination, lora_path = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)
if len(config.get("lora_configs", [])) != 1:
    raise ValueError("The MiniMax-H3 Turbo config must contain exactly one LoRA entry")
config["lora_configs"][0]["path"] = lora_path
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, ensure_ascii=False)
PY

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${lightx2v_path}/lightx2v_kernel_xpu/python:${PYTHONPATH}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

prompt=${PROMPT:-A cinematic fox walking through a snowy forest, with soft wind and distant birds.}
seed=${SEED:-42}

echo "MiniMax-H3 model: ${model_path}"
echo "Turbo LoRA: ${lora_path}"
echo "Output: ${output_path}"

torchrun --standalone --nproc_per_node=1 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${runtime_config}" \
  --prompt "${prompt}" \
  --save_result_path "${output_path}" \
  --seed "${seed}"
