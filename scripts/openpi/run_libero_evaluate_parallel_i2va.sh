#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"

gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
output_root="${OPENPI_PARALLEL_OUTPUT_ROOT:-${lightx2v_path}/save_results/pi05_libero_pytorch_fp32_parallel_evaluation}"

IFS=',' read -ra gpus <<< "${gpu_list}"
suites=(libero_spatial libero_object libero_goal libero_10)
case "${#gpus[@]}" in
  1) assignments=("${suites[*]}") ;;
  2) assignments=("libero_10 libero_spatial" "libero_goal libero_object") ;;
  4) assignments=("${suites[@]}") ;;
  *)
    echo "LIBERO-40 parallel evaluation requires 1, 2, or 4 GPUs" >&2
    exit 2
    ;;
esac

mkdir -p "${output_root}/logs" "${output_root}/runtime"
exec {lock_fd}>"${output_root}/.parallel.lock"
if ! flock -n "${lock_fd}"; then
  echo "Another evaluation is using ${output_root}" >&2
  exit 2
fi
for suite in "${suites[@]}"; do
  rm -f "${output_root}/runtime/${suite}.status"
done

run_worker() {
  local worker_index="$1"
  local gpu="${gpus[worker_index]}"
  local child_pid=""
  local worker_status=0
  local suite suite_output suite_runtime log_path exit_code

  stop_suite() {
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
      kill -TERM -- "-${child_pid}" 2>/dev/null || kill -TERM "${child_pid}" 2>/dev/null || true
      wait "${child_pid}" 2>/dev/null || true
    fi
    exit 143
  }
  trap stop_suite INT TERM

  for suite in ${assignments[worker_index]}; do
    suite_output="${output_root}/${suite}"
    suite_runtime="${output_root}/runtime/${suite}"
    log_path="${output_root}/logs/${suite}.log"
    mkdir -p "${suite_runtime}/libero_config" "${suite_runtime}/numba" "${suite_runtime}/matplotlib" "${suite_runtime}/cache"
    printf '\n[%(%Y-%m-%dT%H:%M:%SZ)T] suite=%s gpu=%s\n' -1 "${suite}" "${gpu}" >> "${log_path}"

    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export OPENPI_EVAL_OUTPUT_DIR="${suite_output}"
      export OPENPI_RUNTIME_DIR="${suite_runtime}"
      export OPENPI_LIBERO_CONFIG_DIR="${suite_runtime}/libero_config"
      export OPENPI_EVAL_BENCHMARKS="${suite}"
      export NUMBA_CACHE_DIR="${suite_runtime}/numba"
      export MPLCONFIGDIR="${suite_runtime}/matplotlib"
      export XDG_CACHE_HOME="${suite_runtime}/cache"
      exec setsid bash "${script_dir}/run_libero_evaluate_i2va.sh"
    ) >> "${log_path}" 2>&1 &
    child_pid=$!
    if wait "${child_pid}"; then
      exit_code=0
    else
      exit_code=$?
      worker_status=1
    fi
    child_pid=""
    printf '%s\n' "${exit_code}" > "${output_root}/runtime/${suite}.status"
    echo "${suite}: exit=${exit_code}, log=${log_path}"
  done
  return "${worker_status}"
}

worker_pids=()
stop_workers() {
  local pid
  for pid in "${worker_pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${worker_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap stop_workers EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

for index in "${!gpus[@]}"; do
  echo "GPU ${gpus[index]}: ${assignments[index]}"
  run_worker "${index}" &
  worker_pids+=("$!")
done

worker_failed=0
for pid in "${worker_pids[@]}"; do
  wait "${pid}" || worker_failed=1
done
worker_pids=()
trap - EXIT HUP INT TERM

summary_command=(
  python "${script_dir}/support/libero_summary.py"
  --output-root "${output_root}"
)
for suite in "${suites[@]}"; do
  exit_code=1
  [[ ! -f "${output_root}/runtime/${suite}.status" ]] || read -r exit_code < "${output_root}/runtime/${suite}.status"
  summary_command+=(--worker-status "${suite}=${exit_code}")
done

"${summary_command[@]}" || worker_failed=1
exit "${worker_failed}"
