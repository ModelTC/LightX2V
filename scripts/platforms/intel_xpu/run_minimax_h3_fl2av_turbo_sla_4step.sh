#!/usr/bin/env bash

# AdaLN cache setup:
# If the inference JSON config enables "use_adaln_cache": true, generate the cache before inference:
# 1. Set lightx2v_path, model_path, --config_json, and --model-variant in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh.
# 2. Use --model-variant fl2av for t2av/i2av/l2av/fl2av, or --model-variant ref2av for ref2av.
# 3. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
# Cache generation and inference must use the same JSON config and adaln_cache_dir.
set -euo pipefail

# Usage:
#   # Single XPU (default)
#   bash scripts/platforms/intel_xpu/run_minimax_h3_fl2av_turbo_sla_4step.sh
#
#   # Tensor parallel on 2 XPUs
#   PARALLEL_MODE=tp TP_SIZE=2 \
#     bash scripts/platforms/intel_xpu/run_minimax_h3_fl2av_turbo_sla_4step.sh
#
#   # Sequence parallel on 2 XPUs
#   PARALLEL_MODE=sp SP_SIZE=2 \
#     bash scripts/platforms/intel_xpu/run_minimax_h3_fl2av_turbo_sla_4step.sh
#
#   # Tensor parallel 2 x sequence parallel 2 (4 XPUs)
#   PARALLEL_MODE=sp_tp TP_SIZE=2 SP_SIZE=2 \
#     bash scripts/platforms/intel_xpu/run_minimax_h3_fl2av_turbo_sla_4step.sh
#
# Optional overrides include ZE_AFFINITY_MASK, MODEL_PATH, LORA_PATH,
# CONFIG_JSON, FIRST_FRAME, LAST_FRAME, OUTPUT_PATH, PROMPT, and SEED.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/MiniMax-H3}
lora_path=${LORA_PATH:-/llm/models/Minimax-h3-Turbo-SLA/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_bf16.safetensors}
config_template=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/intel_xpu/minimax_h3_fl2v_turbo_sla_4step.json}
first_frame=${FIRST_FRAME:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png}
last_frame=${LAST_FRAME:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_minimax_h3_fl2av_turbo_sla_4step.mp4}
prompt=${PROMPT:-Create a coherent cinematic transition between the two frames with natural synchronized ambient sound.}
seed=${SEED:-42}
parallel_mode=${PARALLEL_MODE:-single}
tp_size=${TP_SIZE:-2}
sp_size=${SP_SIZE:-2}

case "${parallel_mode}" in
  single) tp_size=1; sp_size=1 ;;
  tp) sp_size=1 ;;
  sp) tp_size=1 ;;
  sp_tp) ;;
  *) echo "PARALLEL_MODE must be one of: single, tp, sp, sp_tp" >&2; exit 1 ;;
esac
nproc_per_node=$((tp_size * sp_size))
if ((tp_size < 1 || sp_size < 1)); then
  echo "TP_SIZE and SP_SIZE must be positive integers" >&2
  exit 1
fi

if [[ -z "${ZE_AFFINITY_MASK:-}" ]]; then
  ZE_AFFINITY_MASK=$(seq -s, 0 $((nproc_per_node - 1)))
  export ZE_AFFINITY_MASK
fi
export PLATFORM=${PLATFORM:-intel_xpu}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONPATH=${PYTHONPATH:-}

if ((sp_size > 1)); then
  # oneCCL settings used by the Ulysses all-to-all path on Intel XPU.
  export CCL_SYCL_ALLTOALL_ARC_LL=${CCL_SYCL_ALLTOALL_ARC_LL:-1}
  export CCL_SYCL_ALLTOALL_TMP_BUF=${CCL_SYCL_ALLTOALL_TMP_BUF:-1}
  export CCL_SYCL_CCL_BARRIER=${CCL_SYCL_CCL_BARRIER:-1}
  export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=${CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD:-4294967296}
  export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=${CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD:-4294967296}
  export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=${CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD:-4294967296}
fi

[[ -d "${model_path}" ]] || { echo "Model directory not found: ${model_path}" >&2; exit 1; }
[[ -f "${lora_path}" ]] || { echo "SLA LoRA checkpoint not found: ${lora_path}" >&2; exit 1; }
[[ -f "${config_template}" ]] || { echo "Config file not found: ${config_template}" >&2; exit 1; }
[[ -f "${first_frame}" ]] || { echo "First frame not found: ${first_frame}" >&2; exit 1; }
[[ -f "${last_frame}" ]] || { echo "Last frame not found: ${last_frame}" >&2; exit 1; }

mkdir -p "$(dirname -- "${output_path}")"
runtime_config=$(mktemp "${TMPDIR:-/tmp}/lightx2v-minimax-h3-sla-XXXXXX.json")
trap 'rm -f -- "${runtime_config}"' EXIT

# Keep the checked-in config reusable while allowing LORA_PATH to override the
# local checkpoint location without editing JSON.
python - "${config_template}" "${runtime_config}" "${lora_path}" "${tp_size}" "${sp_size}" <<'PY'
import json
import sys

source, destination, lora_path, tp_size, sp_size = sys.argv[1:]
tp_size, sp_size = int(tp_size), int(sp_size)
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)
if config.get("attn_type") != "dynamic_sparse_attn":
    raise ValueError("MiniMax-H3 SLA config must use attn_type=dynamic_sparse_attn")
settings = config.get("dynamic_sparse_attn_setting", {})
if settings.get("operator") != "intel_xpu_cute_attn":
    raise ValueError("MiniMax-H3 SLA config must use operator=intel_xpu_cute_attn")
if len(config.get("lora_configs", [])) != 1:
    raise ValueError("MiniMax-H3 SLA config must contain exactly one LoRA entry")
config["lora_configs"][0]["path"] = lora_path
if tp_size > 1 or sp_size > 1:
    parallel = {"tensor_p_size": tp_size, "seq_p_size": sp_size}
    if sp_size > 1:
        parallel.update({"seq_p_attn_type": "ulysses", "seq_p_a2a_backend": "torch"})
    config["parallel"] = parallel
else:
    config.pop("parallel", None)
if tp_size > 1:
    config["tp_mm_type"] = "IntelTensorParallel"
    config["text_encoder_tensor_parallel"] = True
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, ensure_ascii=False)
PY

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

# Fail before loading the 166 GB base model if an older sycl-kernels wheel is
# active or the SLA API was not packaged.
python - <<'PY'
import sycl_kernels

required = ("sla_block_map", "sparse_block_attention", "sla_sparse_attention")
missing = [name for name in required if not callable(getattr(sycl_kernels, name, None))]
if missing:
    raise RuntimeError(f"Installed sycl_kernels is missing SLA APIs: {missing}")
print(f"Using sycl_kernels from {sycl_kernels.__file__}")
PY

echo "MiniMax-H3 model: ${model_path}"
echo "SLA LoRA: ${lora_path}"
echo "Config: ${config_template}"
echo "XPU: ${ZE_AFFINITY_MASK}"
echo "Parallel mode: ${parallel_mode} (TP=${tp_size}, SP=${sp_size}, processes=${nproc_per_node})"
echo "First frame: ${first_frame}"
echo "Last frame: ${last_frame}"
echo "Output: ${output_path}"

torchrun --standalone --nproc_per_node="${nproc_per_node}" -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task fl2av \
  --model_path "${model_path}" \
  --config_json "${runtime_config}" \
  --prompt "${prompt}" \
  --image_path "${first_frame}" \
  --last_frame_path "${last_frame}" \
  --save_result_path "${output_path}" \
  --seed "${seed}"
