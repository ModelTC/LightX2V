#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"
workspace_root="$(dirname -- "${lightx2v_path}")"
openpi_data_root="${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}"

model_path="${OPENPI_MODEL_PATH:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero_pytorch_fp32}"
config_json="${OPENPI_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero.json}"
eval_config="${OPENPI_EVAL_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero_eval.json}"
libero_root="${OPENPI_LIBERO_ROOT:-${workspace_root}/openpi/third_party/libero}"
transformers_runtime="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-${openpi_data_root}/python_deps/openpi_official_pytorch_runtime}"
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
suite_list="libero_spatial,libero_object,libero_goal,libero_10"
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
      export MUJOCO_GL="${MUJOCO_GL:-egl}"
      export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
      export OPENPI_RUN_MODE=evaluate
      export OPENPI_LIBERO_ROOT="${libero_root}"
      export OPENPI_LIBERO_CONFIG_DIR="${suite_runtime}/libero_config"
      export OPENPI_EVAL_CONFIG="${eval_config}"
      export OPENPI_EVAL_BENCHMARKS="${suite}"
      export OPENPI_TRANSFORMERS_RUNTIME_PATH="${transformers_runtime}"
      export NUMBA_CACHE_DIR="${suite_runtime}/numba"
      export MPLCONFIGDIR="${suite_runtime}/matplotlib"
      export XDG_CACHE_HOME="${suite_runtime}/cache"
      export PYTHONNOUSERSITE=1
      export TOKENIZERS_PARALLELISM=false
      export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-0}"
      exec setsid python -m lightx2v.infer \
        --model_cls openpi \
        --task i2va \
        --model_path "${model_path}" \
        --config_json "${config_json}" \
        --seed "${OPENPI_POLICY_SEED:-0}" \
        --save_result_path "${suite_output}"
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
  python "${script_dir}/libero_summary.py"
  --output-root "${output_root}"
  --suites "${suite_list}"
)
for suite in "${suites[@]}"; do
  exit_code=1
  [[ ! -f "${output_root}/runtime/${suite}.status" ]] || read -r exit_code < "${output_root}/runtime/${suite}.status"
  summary_command+=(--worker-status "${suite}=${exit_code}")
done

"${summary_command[@]}" || worker_failed=1
exit "${worker_failed}"
