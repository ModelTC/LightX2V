#!/bin/bash

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
model_path=${lightx2v_path}/models/lightx2v/Wan2.2-Distill-Models

# base.sh expects PYTHONPATH to be defined under `set -u`.
export PYTHONPATH=${PYTHONPATH:-}

source ${lightx2v_path}/scripts/base/base.sh

# Ensure stale disagg services/ports from previous runs do not block bootstrap.
bash ${lightx2v_path}/scripts/disagg/kill_service.sh || true

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
export LOAD_FROM_USER=${LOAD_FROM_USER:-0}

controller_cfg=${lightx2v_path}/configs/disagg/wan22_i2v_distill_controller.json
seed=${SEED:-42}
prompt=${PROMPT:-"Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."}
negative_prompt=${NEGATIVE_PROMPT:-"镜头晃动，色调艳丽，过曝，静态"}
save_result_path=${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/wan22_i2v_dynamic.mp4}

controller_log=${lightx2v_path}/save_results/disagg_wan22_i2v_dynamic_controller.log
user_log=${lightx2v_path}/save_results/disagg_wan22_i2v_dynamic_user.log

controller_wait_timeout_s=${CONTROLLER_WAIT_TIMEOUT_S:-3000}
controller_poll_interval_s=${CONTROLLER_POLL_INTERVAL_S:-5}
fatal_watch_interval_s=${FATAL_WATCH_INTERVAL_S:-2}
fatal_flag_file=${lightx2v_path}/save_results/disagg_wan22_i2v_dynamic_fatal.flag

rm -f "${fatal_flag_file}"

has_fatal_log_error() {
    local log_path="$1"
    [[ -f "${log_path}" ]] || return 1

    # Fail-fast on known fatal patterns so we do not wait for full run completion.
    rg -q "KeyError: '/psm_|resource_tracker: There appear to be [0-9]+ leaked shared_memory objects|Failed to process request for room=|Data(Sender|Receiver) transfer failed for room=" "${log_path}"
}

start_fatal_watchdog() {
    (
        while true; do
            if [[ -f "${fatal_flag_file}" ]]; then
                exit 0
            fi
            if [[ -n "${controller_pid:-}" ]] && ! kill -0 "${controller_pid}" 2>/dev/null; then
                exit 0
            fi
            if has_fatal_log_error "${controller_log}" || has_fatal_log_error "${user_log}"; then
                echo "fatal error detected in logs, stopping services immediately"
                : > "${fatal_flag_file}"
                [[ -n "${user_pid:-}" ]] && kill -TERM "${user_pid}" 2>/dev/null || true
                [[ -n "${controller_pid:-}" ]] && kill -TERM "${controller_pid}" 2>/dev/null || true
                # Give controller/sidecars a short grace window to release rooms.
                for _ in $(seq 1 10); do
                    local_alive=0
                    if [[ -n "${user_pid:-}" ]] && kill -0 "${user_pid}" 2>/dev/null; then
                        local_alive=1
                    fi
                    if [[ -n "${controller_pid:-}" ]] && kill -0 "${controller_pid}" 2>/dev/null; then
                        local_alive=1
                    fi
                    if [[ "${local_alive}" -eq 0 ]]; then
                        break
                    fi
                    sleep 0.5
                done
                bash ${lightx2v_path}/scripts/disagg/kill_service.sh || true
                exit 0
            fi
            sleep "${fatal_watch_interval_s}"
        done
    ) &
    watchdog_pid=$!
}

is_controller_stuck() {
    local log_path="$1"
    [[ -f "${log_path}" ]] || return 1

    local tail_block
    tail_block=$(tail -n 240 "${log_path}" 2>/dev/null || true)
    [[ -n "${tail_block}" ]] || return 1

    # Waiting for decoder results, all GPUs idle, and queues still pending => hard-stuck.
    if echo "${tail_block}" | rg -q "Waiting for decoder results" \
        && echo "${tail_block}" | rg -q "queue_total_pending': [1-9]" \
        && ! echo "${tail_block}" | rg -q "gpu_utilization': ([1-9][0-9]*|0\\.[1-9])"; then
        return 0
    fi
    return 1
}

cleanup() {
    local pids=("${user_pid:-}" "${controller_pid:-}")
    for pid in "${pids[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    if [[ -n "${watchdog_pid:-}" ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
        kill "${watchdog_pid}" 2>/dev/null || true
    fi
    bash ${lightx2v_path}/scripts/disagg/kill_service.sh || true
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

if [[ "${LOAD_FROM_USER}" != "0" ]]; then
    python -m lightx2v.disagg.examples.run_user \
        --controller_host "${DISAGG_CONTROLLER_HOST}" \
        --controller_request_port "${DISAGG_CONTROLLER_REQUEST_PORT}" \
        > ${user_log} 2>&1 &
    user_pid=$!
    echo "run_user started pid=${user_pid}"
else
    echo "LOAD_FROM_USER=${LOAD_FROM_USER}, skip starting run_user"
fi

start_fatal_watchdog

if [[ -n "${user_pid:-}" ]]; then
    wait ${user_pid} || true
    echo "run_user finished"
fi

if [[ -f "${fatal_flag_file}" ]]; then
    echo "fatal error handled by watchdog, exiting early"
    wait "${controller_pid}" 2>/dev/null || true
    exit 125
fi

controller_wait_start=$(date +%s)
while kill -0 "${controller_pid}" 2>/dev/null; do
    now_ts=$(date +%s)
    elapsed=$((now_ts - controller_wait_start))

    if (( elapsed >= controller_wait_timeout_s )); then
        if is_controller_stuck "${controller_log}"; then
            echo "controller stuck detected (all GPUs idle with pending queues), force killing services"
        else
            echo "controller wait timeout (${controller_wait_timeout_s}s), force killing services"
        fi
        kill "${controller_pid}" 2>/dev/null || true
        bash ${lightx2v_path}/scripts/disagg/kill_service.sh || true
        wait "${controller_pid}" 2>/dev/null || true
        exit 124
    fi

    if [[ -f "${fatal_flag_file}" ]]; then
        echo "fatal error handled by watchdog, exiting early"
        wait "${controller_pid}" 2>/dev/null || true
        exit 125
    fi

    sleep "${controller_poll_interval_s}"
done

wait ${controller_pid}
if [[ -n "${watchdog_pid:-}" ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
    kill "${watchdog_pid}" 2>/dev/null || true
fi
echo "controller finished"

echo "logs: ${controller_log} ${user_log}"
