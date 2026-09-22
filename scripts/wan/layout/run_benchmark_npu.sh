#!/bin/bash
set -e

lightx2v_path="$(cd "$(dirname "$0")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/models/Wan2.1-I2V-14B-720P}"
mode="${1:-copy}"
if [ "$#" -gt 0 ]; then shift; fi

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL=0
cd "${lightx2v_path}"

python scripts/wan/layout/benchmark.py "$mode" \
  --model-path "$model_path" \
  --checkpoint "${CHECKPOINT_PATH:-${model_path}/fp8}" "$@"
