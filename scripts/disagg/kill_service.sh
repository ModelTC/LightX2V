#!/bin/bash

set -euo pipefail

SCRIPT_NAME="run_wan_t2v_service.sh"
PORTS=(7788 7789 7790 12788 12789 12790 17788 17789 17790 27788 27789 27790)

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
