#!/bin/bash

set -euo pipefail

lightx2v_path=/root/zht/LightX2V
model_path=${lightx2v_path}/models/lightx2v/Wan2.2-Distill-Models

# base.sh expects PYTHONPATH to be defined under `set -u`.
export PYTHONPATH=${PYTHONPATH:-}

source ${lightx2v_path}/scripts/base/base.sh

disagg_conda_env=${DISAGG_CONDA_ENV:-lightx2v}
if [[ "${DISAGG_SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${disagg_conda_env}" ]]; then
        if ! command -v conda >/dev/null 2>&1; then
            echo "ERROR: conda is not available, cannot activate env ${disagg_conda_env}" >&2
            exit 2
        fi
        set +u
        eval "$(conda shell.bash hook)"
        conda activate "${disagg_conda_env}"
        set -u
        echo "activated conda env: ${disagg_conda_env}"
    fi
fi

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

topology=${DISAGG_TOPOLOGY:-multi_node}
default_controller_cfg=${lightx2v_path}/configs/disagg/multi_node/wan22_i2v_distill_controller.json
if [[ "${topology}" == "single_node" ]]; then
    default_controller_cfg=${lightx2v_path}/configs/disagg/single_node/wan22_i2v_distill_controller.json
fi
controller_cfg=${DISAGG_CONTROLLER_CFG:-${default_controller_cfg}}
if [[ ! -f "${controller_cfg}" ]]; then
    echo "ERROR: controller config not found: ${controller_cfg}" >&2
    exit 2
fi

derived_controller_host=""
if command -v jq >/dev/null 2>&1; then
    derived_controller_host=$(jq -r '.disagg_config.bootstrap_addr // empty' "${controller_cfg}")
fi
export DISAGG_CONTROLLER_HOST=${DISAGG_CONTROLLER_HOST:-${derived_controller_host:-127.0.0.1}}
export DISAGG_CONTROLLER_REQUEST_PORT=${DISAGG_CONTROLLER_REQUEST_PORT:-12786}
export LOAD_FROM_USER=${LOAD_FROM_USER:-0}
export DISAGG_INSTANCE_START_TIMEOUT_SECONDS=${DISAGG_INSTANCE_START_TIMEOUT_SECONDS:-90}
# Dynamic debug defaults to a smaller request batch; override for stress runs.
export DISAGG_AUTO_REQUEST_COUNT=${DISAGG_AUTO_REQUEST_COUNT:-1}
user_start_delay_s=${USER_START_DELAY_S:-0}
if [[ -n "${USER_MAX_REQUESTS:-}" ]]; then
    user_max_requests=${USER_MAX_REQUESTS}
elif [[ "${LOAD_FROM_USER}" != "0" ]]; then
    # When the workload is driven from the user process, keep sending until the stage ends
    # unless the caller explicitly sets a hard cap.
    user_max_requests=0
else
    user_max_requests=${DISAGG_AUTO_REQUEST_COUNT}
fi

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
remote_log_collect=${REMOTE_LOG_COLLECT:-1}
remote_log_collect_dir=${REMOTE_LOG_COLLECT_DIR:-${lightx2v_path}/save_results/remote_logs}
remote_logs_collected=0
remote_pre_clean=${DISAGG_REMOTE_PRE_CLEAN:-1}
is_single_node=0
if [[ "${topology}" == "single_node" ]]; then
    is_single_node=1
fi

echo "disagg topology=${topology}"
echo "controller_cfg=${controller_cfg}"
echo "DISAGG_CONTROLLER_HOST=${DISAGG_CONTROLLER_HOST} DISAGG_CONTROLLER_REQUEST_PORT=${DISAGG_CONTROLLER_REQUEST_PORT}"
echo "DISAGG_AUTO_REQUEST_COUNT=${DISAGG_AUTO_REQUEST_COUNT}"
echo "LOAD_FROM_USER=${LOAD_FROM_USER} USER_START_DELAY_S=${user_start_delay_s} USER_MAX_REQUESTS=${user_max_requests}"

rm -f "${fatal_flag_file}"

pre_clean_remote_hosts_once() {
    if [[ "${is_single_node}" == "1" ]]; then
        echo "skip remote pre-clean: single_node topology"
        return 0
    fi
    if [[ "${remote_pre_clean}" == "0" || "${remote_pre_clean}" == "false" ]]; then
        return 0
    fi
    if ! command -v jq >/dev/null 2>&1; then
        echo "skip remote pre-clean: jq not found"
        return 0
    fi

    local bootstrap_host
    bootstrap_host=$(jq -r '.disagg_config.bootstrap_addr // empty' "${controller_cfg}")
    local ssh_user
    ssh_user=$(jq -r '.disagg_config.ssh_user // empty' "${controller_cfg}")
    local remote_workdir
    remote_workdir=$(jq -r '.disagg_config.remote_workdir // empty' "${controller_cfg}")
    if [[ -z "${remote_workdir}" ]]; then
        remote_workdir="${lightx2v_path}"
    fi

    mapfile -t remote_hosts < <(jq -r --arg bootstrap "${bootstrap_host}" '.disagg_config.static_instance_slots[]?.host // empty | select(length > 0 and . != $bootstrap)' "${controller_cfg}" | sort -u)
    if (( ${#remote_hosts[@]} == 0 )); then
        echo "no remote hosts discovered for pre-clean"
        return 0
    fi

    mapfile -t ssh_opts < <(jq -r '.disagg_config.ssh_options[]? // empty' "${controller_cfg}")

    local remote_workdir_q
    remote_workdir_q=$(printf '%q' "${remote_workdir}")
    local remote_cmd
    remote_cmd="set -e; cd ${remote_workdir_q}; bash scripts/disagg/kill_service.sh || true"

    for host in "${remote_hosts[@]}"; do
        local target="${host}"
        if [[ -n "${ssh_user}" ]]; then
            target="${ssh_user}@${host}"
        fi

        echo "remote pre-clean on ${host}"
        if ssh "${ssh_opts[@]}" "${target}" "bash -lc '${remote_cmd}'"; then
            echo "remote pre-clean succeeded on ${host}"
        else
            echo "warning: remote pre-clean failed on ${host}"
        fi
    done
}

sync_remote_configs_once() {
    if [[ "${is_single_node}" == "1" ]]; then
        echo "skip remote config sync: single_node topology"
        return 0
    fi
    if ! command -v jq >/dev/null 2>&1; then
        echo "skip remote config sync: jq not found"
        return 0
    fi
    if ! command -v scp >/dev/null 2>&1; then
        echo "skip remote config sync: scp not found"
        return 0
    fi

    local bootstrap_host
    bootstrap_host=$(jq -r '.disagg_config.bootstrap_addr // empty' "${controller_cfg}")
    local ssh_user
    ssh_user=$(jq -r '.disagg_config.ssh_user // empty' "${controller_cfg}")
    local remote_workdir
    remote_workdir=$(jq -r '.disagg_config.remote_workdir // empty' "${controller_cfg}")
    if [[ -z "${remote_workdir}" ]]; then
        remote_workdir="${lightx2v_path}"
    fi

    mapfile -t remote_hosts < <(jq -r --arg bootstrap "${bootstrap_host}" '.disagg_config.static_instance_slots[]?.host // empty | select(length > 0 and . != $bootstrap)' "${controller_cfg}" | sort -u)
    if (( ${#remote_hosts[@]} == 0 )); then
        echo "no remote hosts discovered for config sync"
        return 0
    fi

    mapfile -t ssh_opts < <(jq -r '.disagg_config.ssh_options[]? // empty' "${controller_cfg}")

    local config_files=("${controller_cfg}")
    for role in encoder transformer decoder; do
        local cfg_candidate="${controller_cfg/_controller.json/_${role}.json}"
        if [[ -f "${cfg_candidate}" ]]; then
            config_files+=("${cfg_candidate}")
        fi
    done

    for host in "${remote_hosts[@]}"; do
        local target="${host}"
        if [[ -n "${ssh_user}" ]]; then
            target="${ssh_user}@${host}"
        fi

        for src_cfg in "${config_files[@]}"; do
            local rel_cfg="${src_cfg#${lightx2v_path}/}"
            local dst_cfg="${src_cfg}"
            if [[ "${src_cfg}" == "${lightx2v_path}/"* ]]; then
                dst_cfg="${remote_workdir}/${rel_cfg}"
            fi

            local dst_dir
            dst_dir=$(dirname "${dst_cfg}")
            ssh "${ssh_opts[@]}" "${target}" "mkdir -p '${dst_dir}'" || true
            if scp "${ssh_opts[@]}" "${src_cfg}" "${target}:${dst_cfg}" >/dev/null 2>&1; then
                echo "synced config to ${host}:${dst_cfg}"
            else
                echo "warning: failed to sync config to ${host}:${dst_cfg}"
            fi
        done
    done
}

collect_remote_logs_once() {
    if [[ "${is_single_node}" == "1" ]]; then
        echo "skip remote log collection: single_node topology"
        return 0
    fi
    if [[ "${remote_log_collect}" == "0" || "${remote_log_collect}" == "false" ]]; then
        return 0
    fi
    if [[ "${remote_logs_collected}" == "1" ]]; then
        return 0
    fi
    remote_logs_collected=1

    if [[ ! -f "${controller_cfg}" ]]; then
        echo "skip remote log collection: controller config not found: ${controller_cfg}"
        return 0
    fi
    if ! command -v jq >/dev/null 2>&1; then
        echo "skip remote log collection: jq not found"
        return 0
    fi

    local remote_log_dir
    remote_log_dir=$(jq -r '.disagg_config.remote_log_dir // empty' "${controller_cfg}")
    local bootstrap_host
    bootstrap_host=$(jq -r '.disagg_config.bootstrap_addr // empty' "${controller_cfg}")
    local ssh_user
    ssh_user=$(jq -r '.disagg_config.ssh_user // empty' "${controller_cfg}")
    if [[ -z "${remote_log_dir}" ]]; then
        echo "skip remote log collection: disagg_config.remote_log_dir is empty"
        return 0
    fi

    mapfile -t remote_hosts < <(jq -r --arg bootstrap "${bootstrap_host}" '.disagg_config.static_instance_slots[]?.host // empty | select(length > 0 and . != $bootstrap)' "${controller_cfg}" | sort -u)
    if (( ${#remote_hosts[@]} == 0 )); then
        echo "no remote hosts discovered from static_instance_slots, skip remote log collection"
        return 0
    fi

    mapfile -t ssh_opts < <(jq -r '.disagg_config.ssh_options[]? // empty' "${controller_cfg}")

    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${remote_log_collect_dir}"

    for host in "${remote_hosts[@]}"; do
        local target="${host}"
        if [[ -n "${ssh_user}" ]]; then
            target="${ssh_user}@${host}"
        fi

        local dest_dir="${remote_log_collect_dir}/${host}_${ts}"
        local archive_path="${dest_dir}/remote_logs.tgz"
        mkdir -p "${dest_dir}"

        local remote_log_dir_q
        remote_log_dir_q=$(printf '%q' "${remote_log_dir}")
        local remote_cmd
        remote_cmd="set -e; shopt -s nullglob; cd ${remote_log_dir_q}; files=(*_service.log *_sidecar.log); if (( \${#files[@]} == 0 )); then exit 3; fi; tar -czf - -- \"\${files[@]}\""

        if ssh "${ssh_opts[@]}" "${target}" "bash -lc '${remote_cmd}'" > "${archive_path}" 2>/dev/null; then
            tar -xzf "${archive_path}" -C "${dest_dir}" >/dev/null 2>&1 || true
            rm -f "${archive_path}"
            echo "remote logs collected from ${host} -> ${dest_dir}"
        else
            rm -f "${archive_path}"
            echo "warning: failed to collect remote logs from ${host}:${remote_log_dir}"
        fi
    done
}

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
    collect_remote_logs_once || true
}

trap cleanup EXIT INT TERM

pre_clean_remote_hosts_once
sync_remote_configs_once

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
    if [[ "${user_start_delay_s}" != "0" ]]; then
        echo "waiting ${user_start_delay_s}s before run_user to let remote services warm up"
        sleep "${user_start_delay_s}"
    fi
    python -m lightx2v.disagg.examples.run_user \
        --controller_host "${DISAGG_CONTROLLER_HOST}" \
        --controller_request_port "${DISAGG_CONTROLLER_REQUEST_PORT}" \
        --max_requests "${user_max_requests}" \
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
