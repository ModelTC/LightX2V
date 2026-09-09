#!/usr/bin/env bash

# AdaLN cache setup:
# If the inference JSON config enables "use_adaln_cache": true, generate the cache before inference:
# 1. Set lightx2v_path, model_path, --config_json, and --task in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh.
# 2. Use --task fl2av for t2av/i2av/l2av/fl2av, or --task ref2av for ref2av.
# 3. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
# Cache generation and inference must use the same JSON config and adaln_cache_dir.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/MiniMax-H3}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/intel_xpu/minimax_h3_t2av_int8.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_int8.mp4}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export PLATFORM=${PLATFORM:-intel_xpu}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH=${PYTHONPATH:-}

mkdir -p "$(dirname -- "${output_path}")"

[[ -d "${model_path}" ]] || { echo "Model directory not found: ${model_path}"; exit 1; }
[[ -f "${config_json}" ]] || { echo "Config file not found: ${config_json}"; exit 1; }
source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${lightx2v_path}/lightx2v_kernel_xpu/python:${PYTHONPATH}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
echo "Effective dtype overrides: DTYPE=${DTYPE}, SENSITIVE_LAYER_DTYPE=${SENSITIVE_LAYER_DTYPE}"

prompt=${PROMPT:-A cinematic fox walking through a snowy forest}
seed=${SEED:-42}

torchrun --standalone --nproc_per_node=1 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${prompt}" \
  --save_result_path "${output_path}" \
  --seed "${seed}"
