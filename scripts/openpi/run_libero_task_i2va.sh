#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/openpi/run_libero_task_i2va.sh <benchmark> <task_id> [init_state_id]

Example:
  bash scripts/openpi/run_libero_task_i2va.sh libero_goal 3 0

Benchmarks: libero_spatial, libero_object, libero_goal, libero_10, libero_90
IDs are zero-based; init_state_id defaults to 0.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

benchmark="$1"
task_id="$2"
init_state_id="${3:-0}"

result_root="${OPENPI_LIBERO_RESULT_ROOT:-${lightx2v_path}/save_results/openpi_libero_tasks}"
task_output_dir="${result_root}/${benchmark}_task_${task_id}"
output_prefix="init_state_${init_state_id}"

LIBERO_BENCHMARK="${benchmark}" \
LIBERO_TASK_ID="${task_id}" \
LIBERO_INIT_STATE_ID="${init_state_id}" \
OPENPI_SAVE_VIDEO_PATH="${task_output_dir}/${output_prefix}.mp4" \
OPENPI_SAVE_ACTION_PATH="${task_output_dir}/${output_prefix}.actions.npy" \
OPENPI_SAVE_METRICS_PATH="${task_output_dir}/${output_prefix}.metrics.json" \
bash "${lightx2v_path}/scripts/openpi/run_libero_i2va.sh"
