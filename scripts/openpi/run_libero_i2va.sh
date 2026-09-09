#!/usr/bin/env bash

set -euo pipefail

lightx2v_path="${LIGHTX2V_PATH:-/data/liuhongda/LightX2V}"
model_path="${OPENPI_MODEL_PATH:-/data/liuhongda/openpi_data/openpi-assets/checkpoints/pi05_libero_pytorch}"
config_json="${OPENPI_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero.json}"
transformers_runtime_path="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-/data/liuhongda/openpi_data/python_deps/openpi_pytorch_runtime}"
python_bin="${OPENPI_PYTHON:-/opt/conda/bin/python}"
run_mode="${OPENPI_RUN_MODE:-rollout}"

image_path="${OPENPI_IMAGE_PATH:-/data/liuhongda/openpi_data/examples/pi05_libero/libero_spatial_task0_demo0_step0}"
state_path="${OPENPI_STATE_PATH:-${image_path}/state.npy}"
task_description="${OPENPI_TASK_DESCRIPTION:-}"
save_action_path="${OPENPI_SAVE_ACTION_PATH:-${lightx2v_path}/save_results/output_openpi_pi05_libero.actions.npy}"
save_video_path="${OPENPI_SAVE_VIDEO_PATH:-${OPENPI_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/output_openpi_pi05_libero.mp4}}"
save_metrics_path="${OPENPI_SAVE_METRICS_PATH:-${lightx2v_path}/save_results/output_openpi_pi05_libero.metrics.json}"

libero_root="${OPENPI_LIBERO_ROOT:-/data/liuhongda/openpi/third_party/libero}"
libero_config_dir="${OPENPI_LIBERO_CONFIG_DIR:-/data/liuhongda/openpi_data/runtime_configs/lightx2v_openpi_libero}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}"
export USE_FLAX=0
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${OPENPI_EGL_DEVICE_ID:-0}"
export NUMBA_CACHE_DIR="${OPENPI_NUMBA_CACHE_DIR:-${NUMBA_CACHE_DIR:-/tmp/lightx2v-openpi-numba-cache}}"
export MPLCONFIGDIR="${OPENPI_MPLCONFIG_DIR:-${MPLCONFIGDIR:-/tmp/lightx2v-openpi-matplotlib-cache}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

for required in "${python_bin}" "${model_path}/model.safetensors" "${config_json}" "${transformers_runtime_path}/transformers"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required OpenPI runtime input is missing: ${required}" >&2
    exit 1
  fi
done

source "${lightx2v_path}/scripts/base/base.sh"

# The common LightX2V process stays on the base environment. OpenPIRunner adds
# the isolated patched Transformers runtime only to its local worker process.
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENPI_PYTHON="${python_bin}"
export OPENPI_RUN_MODE="${run_mode}"
export OPENPI_TRANSFORMERS_RUNTIME_PATH="${transformers_runtime_path}"
export OPENPI_LIBERO_ROOT="${libero_root}"
export OPENPI_LIBERO_CONFIG_DIR="${libero_config_dir}"
export OPENPI_SAVE_METRICS_PATH="${save_metrics_path}"
unset OPENPI_WORKER_PROCESS

case "${run_mode}" in
  rollout)
    seed="${OPENPI_SEED:-7}"
    if [[ ! -d "${libero_root}/libero/libero/bddl_files" ]]; then
      echo "LIBERO checkout is incomplete: ${libero_root}" >&2
      exit 1
    fi
    ;;
  single_observation)
    seed="${OPENPI_SEED:-0}"
    if [[ -z "${task_description}" ]]; then
      task_description="pick up the black bowl between the plate and the ramekin and place it on the plate"
    fi
    ;;
  *)
    echo "Unsupported OPENPI_RUN_MODE=${run_mode}; expected rollout or single_observation." >&2
    exit 2
    ;;
esac

echo "OpenPI call chain: lightx2v.infer -> OpenPIRunner -> local ${run_mode} worker"
"${python_bin}" -m lightx2v.infer \
  --model_cls openpi \
  --task i2va \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --seed "${seed}" \
  --prompt "${task_description}" \
  --image_path "${image_path}" \
  --state_path "${state_path}" \
  --save_result_path "${save_video_path}" \
  --save_action_path "${save_action_path}"
