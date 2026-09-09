#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Run one task from the local LIBERO BDDL and initialization-state samples.

Usage:
  bash scripts/openpi/run_libero_task_i2va.sh <benchmark> <task_id> [init_state_id]

Benchmarks and zero-based task-id ranges:
  libero_spatial  0-9
  libero_object   0-9
  libero_goal     0-9
  libero_10       0-9
  libero_90       0-89

Example:
  bash scripts/openpi/run_libero_task_i2va.sh libero_goal 3 0

Each task's pruned init file contains 50 states; init_state_id is 0-49.

Optional environment variables:
  OPENPI_LIBERO_RESULT_ROOT  Result root directory
  OPENPI_MAX_STEPS           Override the suite-specific rollout step limit
  OPENPI_SEED                Policy and simulator seed (default: 7)
  OPENPI_VIDEO_FPS           Output video FPS (default: 10)
  OPENPI_RENDER_SIZE         Square render resolution (default: 256)
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

case "${benchmark}" in
  libero_spatial|libero_object|libero_goal|libero_10)
    task_count=10
    ;;
  libero_90)
    task_count=90
    ;;
  *)
    echo "Unsupported LIBERO benchmark: ${benchmark}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! "${task_id}" =~ ^[0-9]+$ ]]; then
  echo "task_id must be a non-negative integer: ${task_id}" >&2
  exit 2
fi
if [[ ! "${init_state_id}" =~ ^[0-9]+$ ]]; then
  echo "init_state_id must be a non-negative integer: ${init_state_id}" >&2
  exit 2
fi

task_id=$((10#${task_id}))
init_state_id=$((10#${init_state_id}))
if (( task_id >= task_count )); then
  echo "task_id ${task_id} is out of range for ${benchmark}; expected 0-$((task_count - 1))." >&2
  exit 2
fi
if (( init_state_id >= 50 )); then
  echo "init_state_id ${init_state_id} is out of range; expected 0-49." >&2
  exit 2
fi

lightx2v_path="${LIGHTX2V_PATH:-/data/liuhongda/LightX2V}"
libero_root="${OPENPI_LIBERO_ROOT:-/data/liuhongda/openpi/third_party/libero}"
bddl_dir="${libero_root}/libero/libero/bddl_files/${benchmark}"
init_dir="${libero_root}/libero/libero/init_files/${benchmark}"

for required_dir in "${bddl_dir}" "${init_dir}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required LIBERO sample directory is missing: ${required_dir}" >&2
    exit 1
  fi
done

result_root="${OPENPI_LIBERO_RESULT_ROOT:-${lightx2v_path}/save_results/openpi_libero_tasks}"
task_output_dir="${result_root}/${benchmark}_task_${task_id}"
output_prefix="init_state_${init_state_id}"
mkdir -p "${task_output_dir}"

echo "LIBERO benchmark : ${benchmark}"
echo "Task ID          : ${task_id} (zero-based)"
echo "Init-state ID    : ${init_state_id} (zero-based)"
echo "BDDL directory   : ${bddl_dir}"
echo "Init directory   : ${init_dir}"
echo "Output directory : ${task_output_dir}"

LIBERO_BENCHMARK="${benchmark}" \
LIBERO_TASK_ID="${task_id}" \
LIBERO_INIT_STATE_ID="${init_state_id}" \
OPENPI_SAVE_VIDEO_PATH="${task_output_dir}/${output_prefix}.mp4" \
OPENPI_SAVE_ACTION_PATH="${task_output_dir}/${output_prefix}.actions.npy" \
OPENPI_SAVE_METRICS_PATH="${task_output_dir}/${output_prefix}.metrics.json" \
bash "${lightx2v_path}/scripts/openpi/run_libero_i2va.sh"

echo "Saved LIBERO task results under: ${task_output_dir}"
