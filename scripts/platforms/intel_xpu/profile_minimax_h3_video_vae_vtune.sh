#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

model_path=${MODEL_PATH:-/llm/models/MiniMax-H3}
config_json=${CONFIG_JSON:-${REPO_ROOT}/configs/platforms/intel_xpu/minimax_h3_fl2v_turbo_sla_4step.json}
collection=${VTUNE_COLLECTION:-gpu-hotspots}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
result_dir=${VTUNE_RESULT_DIR:-${REPO_ROOT}/logs/vtune-minimax-h3-vae-${timestamp}}
log_path=${VTUNE_LOG_PATH:-${result_dir}.log}

if ! command -v vtune >/dev/null 2>&1; then
  if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    # VTune is commonly added to PATH by the oneAPI environment setup.
    set +u
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
    set -u
  fi
fi

command -v vtune >/dev/null 2>&1 || {
  echo "VTune CLI not found. Install Intel VTune or source the oneAPI environment." >&2
  exit 1
}
[[ -d "${model_path}" ]] || { echo "Model directory not found: ${model_path}" >&2; exit 1; }
[[ -f "${config_json}" ]] || { echo "Config file not found: ${config_json}" >&2; exit 1; }
[[ ! -e "${result_dir}" ]] || { echo "VTune result path already exists: ${result_dir}" >&2; exit 1; }

mkdir -p "$(dirname -- "${result_dir}")" "$(dirname -- "${log_path}")"

export PLATFORM=${PLATFORM:-intel_xpu}
export DTYPE=${DTYPE:-BF16}
export SENSITIVE_LAYER_DTYPE=${SENSITIVE_LAYER_DTYPE:-BF16}
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export PYTHONPATH=${REPO_ROOT}/lightx2v_kernel_xpu/python:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUNBUFFERED=1
# Avoid short-lived TorchInductor worker processes confusing VTune's
# instrumentation control channel during torch.compile warmup.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

vae_args=(
  "${REPO_ROOT}/tools/profile_minimax_h3_video_vae.py"
  --model-path "${model_path}"
  --config-json "${config_json}"
  --warmup "${WARMUP:-0}"
  --iterations "${ITERATIONS:-1}"
)

[[ -n "${FRAMES:-}" ]] && vae_args+=(--frames "${FRAMES}")
[[ -n "${HEIGHT:-}" ]] && vae_args+=(--height "${HEIGHT}")
[[ -n "${WIDTH:-}" ]] && vae_args+=(--width "${WIDTH}")
if [[ -n "${TILE_HEIGHT:-}" || -n "${TILE_WIDTH:-}" ]]; then
  [[ -n "${TILE_HEIGHT:-}" && -n "${TILE_WIDTH:-}" ]] || {
    echo "TILE_HEIGHT and TILE_WIDTH must be set together" >&2
    exit 1
  }
  vae_args+=(--tile-height "${TILE_HEIGHT}" --tile-width "${TILE_WIDTH}")
fi
case "${CPU_OFFLOAD:-config}" in
  1|true|on) vae_args+=(--cpu-offload) ;;
  0|false|off) vae_args+=(--no-cpu-offload) ;;
  config) ;;
  *) echo "CPU_OFFLOAD must be config, 1/true/on, or 0/false/off" >&2; exit 1 ;;
esac

echo "VTune collection: ${collection}"
echo "VTune result: ${result_dir}"
echo "Log: ${log_path}"
echo "XPU: ${ZE_AFFINITY_MASK}"

cd "${REPO_ROOT}"
vtune \
  -collect "${collection}" \
  -no-follow-child \
  -result-dir "${result_dir}" \
  -- python "${vae_args[@]}" \
  2>&1 | tee "${log_path}"

echo "Open result with: vtune-gui ${result_dir}"
