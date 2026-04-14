#!/bin/bash

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
model_path=${lightx2v_path}/models/lightx2v/Wan2.2-Distill-Models

# base.sh expects PYTHONPATH to be defined under `set -u`.
export PYTHONPATH=${PYTHONPATH:-}

source ${lightx2v_path}/scripts/base/base.sh

export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13
if [[ -n "${NVCC_PREPEND_FLAGS:-}" ]]; then
    export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -allow-unsupported-compiler"
else
    export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
fi

export RDMA_IFACE=${RDMA_IFACE:-erdma_0}
export MOONCAKE_DEVICE_NAME=${MOONCAKE_DEVICE_NAME:-eth0}
if [[ -z "${MOONCAKE_LOCAL_HOSTNAME:-}" ]]; then
    _mc_ip=$(ip -4 -o addr show dev "${MOONCAKE_DEVICE_NAME}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1)
    if [[ -n "${_mc_ip}" ]]; then
        export MOONCAKE_LOCAL_HOSTNAME="${_mc_ip}"
    fi
fi

export DISAGG_CONTROLLER_HOST=${DISAGG_CONTROLLER_HOST:-127.0.0.1}
export DISAGG_CONTROLLER_REQUEST_PORT=${DISAGG_CONTROLLER_REQUEST_PORT:-12786}

controller_cfg=${lightx2v_path}/configs/disagg/wan22_i2v_distill_controller.json
seed=${SEED:-42}
prompt=${PROMPT:-"Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."}
negative_prompt=${NEGATIVE_PROMPT:-"镜头晃动，色调艳丽，过曝，静态"}
save_result_path=${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/wan22_i2v_dynamic.mp4}

controller_log=${lightx2v_path}/save_results/disagg_wan22_i2v_dynamic_controller.log
user_log=${lightx2v_path}/save_results/disagg_wan22_i2v_dynamic_user.log

cleanup() {
    local pids=("${user_pid:-}" "${controller_pid:-}")
    for pid in "${pids[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup EXIT INT TERM

python -m lightx2v.disagg.examples.run_service \
    --service controller \
    --model_cls wan2.2_moe \
    --task i2v \
    --model_path ${model_path} \
    --config_json ${controller_cfg} \
    --seed ${seed} \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --save_result_path ${save_result_path} \
    > ${controller_log} 2>&1 &
controller_pid=$!

echo "controller started pid=${controller_pid}"
sleep 8

python -m lightx2v.disagg.examples.run_user \
    --controller_host "${DISAGG_CONTROLLER_HOST}" \
    --controller_request_port "${DISAGG_CONTROLLER_REQUEST_PORT}" \
    > ${user_log} 2>&1 &
user_pid=$!

echo "run_user started pid=${user_pid}"

wait ${user_pid}
echo "run_user finished"

wait ${controller_pid}
echo "controller finished"

echo "logs: ${controller_log} ${user_log}"
