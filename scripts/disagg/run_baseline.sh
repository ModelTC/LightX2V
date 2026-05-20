#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/root/zht/LightX2V}
model_path=${WAN22_MOE_MODEL_PATH:-/root/zht/LightX2V/models/Wan-AI/Wan2.2-I2V-A14B}
config_json=${BASELINE_CONFIG_JSON:-${lightx2v_path}/configs/disagg/baseline/wan22_moe_i2v_baseline.json}

# base.sh expects PYTHONPATH to be defined under `set -u`.
export PYTHONPATH=${PYTHONPATH:-}

baseline_conda_env=${BASELINE_CONDA_ENV:-lightx2v}
if [[ "${BASELINE_SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${baseline_conda_env}" ]]; then
        if ! command -v conda >/dev/null 2>&1; then
            echo "ERROR: conda is not available, cannot activate env ${baseline_conda_env}" >&2
            exit 2
        fi
        set +u
        eval "$(conda shell.bash hook)"
        conda activate "${baseline_conda_env}"
        set -u
        echo "activated conda env: ${baseline_conda_env}"
    fi
fi

if [[ -n "${NUM_GPUS:-}" ]]; then
    num_gpus=${NUM_GPUS}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    num_gpus=$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")
else
    num_gpus=8
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus - 1)))
fi

source "${lightx2v_path}/scripts/base/base.sh"

baseline_log=${BASELINE_LOG:-${lightx2v_path}/save_results/baseline_wan22_i2v_single_task.log}
prompt=${BASELINE_PROMPT:-"Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."}
negative_prompt=${BASELINE_NEGATIVE_PROMPT:-"镜头晃动，色调艳丽，过曝，静态"}
image_path=${BASELINE_IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/img_0.jpg}
save_result_path=${BASELINE_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/output_lightx2v_wan22_moe_i2v_baseline.mp4}
python_executable=${BASELINE_PYTHON_EXECUTABLE:-python}
model_cls=${BASELINE_MODEL_CLS:-wan2.2_moe}
task=${BASELINE_TASK:-i2v}

mkdir -p "${lightx2v_path}/save_results"
exec > "${baseline_log}" 2>&1

gpus_csv="${CUDA_VISIBLE_DEVICES}"
metrics_output_json=${BASELINE_METRICS_OUTPUT_JSON:-${lightx2v_path}/save_results/baseline_controller_metrics.json}
dist_master_addr=${BASELINE_DIST_MASTER_ADDR:-127.0.0.1}
dist_master_port=${BASELINE_DIST_MASTER_PORT:-29600}
request_source=${BASELINE_REQUEST_SOURCE:-generate}
generate_requests=${BASELINE_GENERATE_REQUESTS:-30}
num_workers=${BASELINE_NUM_WORKERS:-1}

"${python_executable}" -m lightx2v.disagg.examples.run_controller \
--mode controller \
--request_source "${request_source}" \
--generate_requests "${generate_requests}" \
--num_workers "${num_workers}" \
--gpus "${gpus_csv}" \
--dist_master_addr "${dist_master_addr}" \
--dist_master_port "${dist_master_port}" \
--python_executable "${python_executable}" \
--model_cls "${model_cls}" \
--task "${task}" \
--model_path "${model_path}" \
--base_config_json "${config_json}" \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--image_path "${image_path}" \
--save_result_path "${save_result_path}" \
--save_dir "$(dirname "${save_result_path}")" \
--metrics_output_json "${metrics_output_json}"
