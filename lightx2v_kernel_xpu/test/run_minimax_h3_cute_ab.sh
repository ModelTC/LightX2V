#!/usr/bin/env bash
set -euo pipefail

kernel_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${kernel_dir}/_cmake_build}"

cmake -S "${kernel_dir}" -B "${build_dir}" \
  -DENABLE_CUTE_FMHA=ON \
  -DXPU_TARGET="${XPU_TARGET:-bmg}" \
  ${CUTLASS_SYCL_ROOT:+-DCUTLASS_SYCL_ROOT="${CUTLASS_SYCL_ROOT}"}
cmake --build "${build_dir}" --target cute_fmha_torch cute_fmha_minimax_h3_torch -j "${BUILD_JOBS:-2}"

read -r -a sequence_lengths <<< "${SEQUENCE_LENGTHS:-19292 37726}"
read -r -a head_counts <<< "${HEAD_COUNTS:-56 28 14 7}"

for sequence_length in "${sequence_lengths[@]}"; do
  for heads in "${head_counts[@]}"; do
    common_args=(
      --build-dir "${build_dir}"
      --sequence-length "${sequence_length}"
      --heads "${heads}"
      --iterations "${ITERATIONS:-1}"
    )

    # Separate processes make the default one-shot comparison insensitive to
    # which library happened to run second after GPU clock boosting.
    ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}" \
      python "${kernel_dir}/test/bench_minimax_h3_cute_ab.py" "${common_args[@]}" --variant generic
    ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}" \
      python "${kernel_dir}/test/bench_minimax_h3_cute_ab.py" "${common_args[@]}" --variant optimized

    if [[ "${VERIFY_OUTPUTS:-0}" == "1" ]]; then
      ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}" \
        python "${kernel_dir}/test/bench_minimax_h3_cute_ab.py" "${common_args[@]}" --variant both
    fi
  done
done
