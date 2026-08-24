#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/MiniMax-H3}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/intel_xpu/dist_infer/minimax_h3_t2av_sp_tp.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_sp_tp.mp4}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3,4,5,6,7}
export PLATFORM=${PLATFORM:-intel_xpu}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH=${PYTHONPATH:-}

# Stable oneCCL paths for Ulysses SP all-to-all on Intel XPU.
export CCL_SYCL_ALLTOALL_ARC_LL=${CCL_SYCL_ALLTOALL_ARC_LL:-1}
export CCL_SYCL_ALLTOALL_TMP_BUF=${CCL_SYCL_ALLTOALL_TMP_BUF:-1}
export CCL_SYCL_CCL_BARRIER=${CCL_SYCL_CCL_BARRIER:-1}
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=${CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD:-4294967296}
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=${CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD:-4294967296}
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=${CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD:-4294967296}

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
mkdir -p "$(dirname -- "${output_path}")"

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${PROMPT:-A cinematic fox walking through a snowy forest}" \
  --save_result_path "${output_path}" \
  --seed "${SEED:-42}"
