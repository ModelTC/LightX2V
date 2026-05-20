#!/bin/bash

set -euo pipefail

SCRIPT_NAMES=("run_baseline.sh")

lightx2v_path=${LIGHTX2V_PATH:-/root/zht/LightX2V}
controller_request_port=${BASELINE_CONTROLLER_REQUEST_PORT:-12786}
result_port=${BASELINE_RESULT_PORT:-12787}
worker_base_port=${BASELINE_WORKER_BASE_PORT:-12888}
worker_monitor_base_port=${BASELINE_WORKER_MONITOR_BASE_PORT:-7888}

if [[ -n "${BASELINE_NUM_WORKERS:-}" ]]; then
    num_workers=${BASELINE_NUM_WORKERS}
elif [[ -n "${NUM_GPUS:-}" ]]; then
    num_workers=${NUM_GPUS}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    num_workers=$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")
else
    num_workers=8
fi

if [[ ! "${num_workers}" =~ ^[0-9]+$ ]] || (( num_workers <= 0 )); then
    num_workers=8
fi

declare -a PORTS=(${controller_request_port} ${result_port})
for ((i = 0; i < num_workers; i++)); do
    PORTS+=($((worker_base_port + i)))
    PORTS+=($((worker_monitor_base_port + i)))
done

# Also cover a small tail range in case worker count changed between runs.
for extra in $(seq 0 31); do
    PORTS+=($((worker_base_port + extra)))
    PORTS+=($((worker_monitor_base_port + extra)))
done

mapfile -t PORTS < <(printf '%s\n' "${PORTS[@]}" | awk 'NF && !seen[$0]++ { print $0 }' | sort -n)

declare -a PROTECTED_PIDS=()

collect_protected_pids() {
    local cur="$$"
    while [[ -n "$cur" && "$cur" != "0" ]]; do
        PROTECTED_PIDS+=("$cur")
        local parent
        parent=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ' || true)
        if [[ -z "$parent" || "$parent" == "$cur" ]]; then
            break
        fi
        cur="$parent"
    done
}

is_protected_pid() {
    local target="$1"
    for p in "${PROTECTED_PIDS[@]}"; do
        if [[ "$p" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

kill_pid_gracefully() {
    local pid="$1"
    if [[ -z "$pid" ]]; then
        return
    fi
    if is_protected_pid "$pid"; then
        echo "Skip protected pid=$pid"
        return
    fi
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
}

find_listen_pids_by_port() {
    local port="$1"

    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u || true
        return
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null | awk -v p=":$port" '
            index($4, p) > 0 {
                while (match($0, /pid=[0-9]+/)) {
                    print substr($0, RSTART + 4, RLENGTH - 4)
                    $0 = substr($0, RSTART + RLENGTH)
                }
            }
        ' | sort -u || true
        return
    fi

    if command -v fuser >/dev/null 2>&1; then
        fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u || true
        return
    fi

    echo "No supported tool found to query listening ports (need one of: lsof, ss, fuser)." >&2
}

collect_protected_pids

for script_name in "${SCRIPT_NAMES[@]}"; do
    echo "Stopping script process: ${script_name}"
    script_pids=$(pgrep -f "$script_name" || true)
    if [[ -z "${script_pids}" ]]; then
        echo "No running process found for ${script_name}"
        continue
    fi
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        echo "Killing script pid=$pid"
        kill_pid_gracefully "$pid"
    done <<< "$script_pids"
done

cleanup_patterns=(
    "lightx2v.disagg.examples.run_controller"
    "lightx2v.disagg.examples.infer"
    "python -m lightx2v.disagg.examples.run_controller"
    "python -m lightx2v.disagg.examples.infer"
    "conda run -n lightx2v bash ${lightx2v_path}/scripts/disagg/run_baseline.sh"
)

for pattern in "${cleanup_patterns[@]}"; do
    echo "Stopping processes matching pattern: ${pattern}"
    matched_pids=$(pgrep -f "$pattern" || true)
    if [[ -z "${matched_pids}" ]]; then
        echo "No process matched: ${pattern}"
        continue
    fi
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        echo "Killing matched pid=$pid"
        kill_pid_gracefully "$pid"
    done <<< "$matched_pids"
done

for port in "${PORTS[@]}"; do
    echo "Stopping listeners on port ${port}"
    port_pids=$(find_listen_pids_by_port "$port")
    if [[ -z "${port_pids}" ]]; then
        echo "No listener found on port ${port}"
        continue
    fi

    while read -r pid; do
        [[ -z "$pid" ]] && continue
        echo "Killing pid=$pid on port ${port}"
        kill_pid_gracefully "$pid"
    done <<< "$port_pids"

    remaining=$(find_listen_pids_by_port "$port")
    if [[ -n "${remaining}" ]]; then
        echo "Warning: port ${port} still has listeners: ${remaining}"
    else
        echo "Port ${port} is clear"
    fi
done

# Best effort cleanup for per-request temp dirs created by run_controller workers.
find /tmp -maxdepth 1 -type d -name 'baseline_req_*' -exec rm -rf {} + 2>/dev/null || true

echo "kill_base.sh done."
