#!/bin/bash

set -euo pipefail

SCRIPT_NAMES=("run_wan22_i2v_distill.sh" "run_dynamic.sh")

list_port=(5566 12788 17788 27788)

collect_proxy_ports_from_config() {
    local config_path="$1"

    if [[ -z "$config_path" || ! -f "$config_path" ]]; then
        return 0
    fi
    if ! command -v jq >/dev/null 2>&1; then
        return 0
    fi

    local base_port
    base_port=$(jq -r '.disagg_config.remote_proxy_req_base_port // empty' "$config_path" 2>/dev/null || true)
    if [[ -z "$base_port" || ! "$base_port" =~ ^[0-9]+$ ]]; then
        return 0
    fi

    jq -r '.disagg_config.static_instance_slots[]?.engine_rank // empty' "$config_path" 2>/dev/null | while read -r engine_rank; do
        [[ -z "$engine_rank" ]] && continue
        if [[ "$engine_rank" =~ ^[0-9]+$ ]]; then
            echo $((base_port + engine_rank))
        fi
    done
}

n=30
list_n=($(seq 0 $((n-1))))

PORTS=(5555 12787)

# Monitor ports for autoscaled services are contiguous from 7788.
for p in $(seq 7788 7803); do
    PORTS+=($p)
done

for a in "${list_port[@]}"; do
    for b in "${list_n[@]}"; do
        PORTS+=($((a + b)))
    done
done

proxy_config_candidates=(
    "${DISAGG_CONTROLLER_CFG:-}"
    "/root/zht/LightX2V/configs/disagg/multi_node/wan22_i2v_distill_controller.json"
    "/root/zht/LightX2V/configs/disagg/single_node/wan22_i2v_distill_controller.json"
)
for config_path in "${proxy_config_candidates[@]}"; do
    while read -r proxy_port; do
        [[ -z "$proxy_port" ]] && continue
        PORTS+=($proxy_port)
    done < <(collect_proxy_ports_from_config "$config_path")
done

# Fallback for environments without jq or without a readable config file.
PORTS+=(28000)

mapfile -t PORTS < <(printf '%s\n' "${PORTS[@]}" | awk 'NF && !seen[$0]++ { print $0 }' | sort -n)

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

declare -a PROTECTED_PIDS=()
collect_protected_pids() {
    local cur="$$"
    while [[ -n "$cur" ]] && [[ "$cur" != "0" ]]; do
        PROTECTED_PIDS+=("$cur")
        local parent
        parent=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ' || true)
        if [[ -z "$parent" ]] || [[ "$parent" == "$cur" ]]; then
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

collect_protected_pids

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

for script_name in "${SCRIPT_NAMES[@]}"; do
    echo "Stopping script process: ${script_name}"
    script_pids=$(pgrep -f "$script_name" || true)
    if [[ -n "${script_pids}" ]]; then
        while read -r pid; do
            [[ -z "$pid" ]] && continue
            echo "Killing script pid=$pid"
            kill_pid_gracefully "$pid"
        done <<< "$script_pids"
    else
        echo "No running process found for ${script_name}"
    fi
done

# Fallback cleanup for orphaned disagg service processes.
cleanup_patterns=(
    "lightx2v.disagg.examples.run_service"
    "lightx2v.disagg.examples.run_user"
    "python -m lightx2v.disagg"
    "conda run -n lightx2v bash scripts/disagg/run_dynamic.sh"
    "conda run -n lightx2v bash scripts/disagg/run_wan22_i2v_distill.sh"
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

echo "kill_service.sh done."
