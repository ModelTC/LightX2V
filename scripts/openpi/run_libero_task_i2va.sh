#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/openpi/run_libero_task_i2va.sh [suite] [task_id] [init_state_id]"
  exit 0
fi
if [[ $# -gt 3 ]]; then
  echo "Usage: bash scripts/openpi/run_libero_task_i2va.sh [suite] [task_id] [init_state_id]" >&2
  exit 2
fi

suite="${1:-${LIBERO_BENCHMARK:-libero_spatial}}"
task_id="${2:-${LIBERO_TASK_ID:-0}}"
init_state_id="${3:-${LIBERO_INIT_STATE_ID:-0}}"

case "${suite}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *) echo "Unknown LIBERO suite: ${suite}" >&2; exit 2 ;;
esac
if [[ ! "${task_id}" =~ ^[0-9]+$ || ! "${init_state_id}" =~ ^[0-9]+$ ]]; then
  echo "task_id and init_state_id must be non-negative integers" >&2
  exit 2
fi

printf -v task_dir 'task_%02d' "$((10#${task_id}))"
printf -v init_dir 'init_%02d' "$((10#${init_state_id}))"
result_root="${OPENPI_LIBERO_RESULT_ROOT:-${lightx2v_path}/save_results/openpi_libero_tasks}"
episode_dir="${result_root}/${suite}/${task_dir}/${init_dir}"

export LIBERO_BENCHMARK="${suite}"
export LIBERO_TASK_ID="$((10#${task_id}))"
export LIBERO_INIT_STATE_ID="$((10#${init_state_id}))"
export OPENPI_SAVE_VIDEO_PATH="${episode_dir}/rollout.mp4"
export OPENPI_SAVE_ACTION_PATH="${episode_dir}/actions.npy"
export OPENPI_SAVE_METRICS_PATH="${episode_dir}/metrics.json"
export OPENPI_RUNTIME_DIR="${episode_dir}/runtime"

exec bash "${script_dir}/run_libero_i2va.sh"
