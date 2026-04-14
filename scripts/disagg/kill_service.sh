#!/bin/bash

set -euo pipefail

SCRIPT_NAME="run_wan22_i2v_distill.sh"

list_port=(5566 12788 17788 27788)

n=30
list_n=($(seq 0 $((n-1))))

PORTS=(5555 7788 7789 7790 12787)

for a in "${list_port[@]}"; do
    for b in "${list_n[@]}"; do
        PORTS+=($((a + b)))
    done
done

kill_pid_gracefully() {
    local pid="$1"
    if [[ -z "$pid" ]]; then
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

echo "Stopping script process: ${SCRIPT_NAME}"
script_pids=$(pgrep -f "$SCRIPT_NAME" || true)
if [[ -n "${script_pids}" ]]; then
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        echo "Killing script pid=$pid"
        kill_pid_gracefully "$pid"
    done <<< "$script_pids"
else
    echo "No running process found for ${SCRIPT_NAME}"
fi

# Fallback cleanup for orphaned disagg service processes.
cleanup_patterns=(
    "lightx2v.disagg.examples.run_service"
    "python -m lightx2v.disagg"
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
