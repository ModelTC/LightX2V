#!/usr/bin/env bash
# Own and supervise the eight reward processes on the dedicated reward node.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash lightx2v_rl/scripts/rewards/start_qwen21_reward_pool.sh

Run on the dedicated eight-GPU reward node. Required environment:
  LIGHTX2V_REWARD_HOST   Node address reachable from every training node.

GPU 0..5: UnifiedReward, ports 8792..8797; GPU 6..7: HPSv3, ports 8791/8798.
All servers bind 0.0.0.0. This foreground supervisor waits for all eight servers
and stops its own children if a child fails or the supervisor receives a signal.
Logs and PID files: lightx2v_rl/outputs/reward_pool/<run-id>/
Optional: LIGHTX2V_REWARD_STARTUP_TIMEOUT (seconds, default 1800),
LIGHTX2V_REWARD_PYTHON, LIGHTX2V_UNIFIEDREWARD_PYTHON.
EOF
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if (( $# )); then usage >&2; exit 2; fi
: "${LIGHTX2V_REWARD_HOST:?Set LIGHTX2V_REWARD_HOST to the reward node address reachable by every training node}"

RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_MODULE_DIR="$(cd -- "$RL_SCRIPT_DIR/../.." && pwd)"
cd "$RL_MODULE_DIR/.."
export PYTHONPATH="$RL_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/lightx2v_rl/models/.hf-runtime}"
if [[ "$HF_HOME" != /* ]]; then export HF_HOME="$PWD/$HF_HOME"; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
RL_UR_PYTHON="${LIGHTX2V_UNIFIEDREWARD_PYTHON:-$RL_MODULE_DIR/.venv-unifiedreward/bin/python}"
RL_HPS_PYTHON="${LIGHTX2V_REWARD_PYTHON:-$RL_MODULE_DIR/.venv-reward/bin/python}"
RL_STARTUP_TIMEOUT="${LIGHTX2V_REWARD_STARTUP_TIMEOUT:-1800}"
if [[ ! "$RL_STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'LIGHTX2V_REWARD_STARTUP_TIMEOUT must be a positive integer.\n' >&2; exit 2
fi
for RL_EXECUTABLE in "$RL_UR_PYTHON" "$RL_HPS_PYTHON"; do
    if [[ ! -x "$RL_EXECUTABLE" ]]; then
        printf 'Reward interpreter unavailable: %s\n' "$RL_EXECUTABLE" >&2
        if [[ "$RL_EXECUTABLE" == "$RL_UR_PYTHON" ]]; then
            printf 'On the GPU server, from the repository root run: bash lightx2v_rl/scripts/setup/setup_unifiedreward_env.sh\n' >&2
            printf 'For an existing custom environment, set LIGHTX2V_UNIFIEDREWARD_PYTHON to its executable.\n' >&2
        else
            printf 'On the GPU server, from the repository root run: bash lightx2v_rl/scripts/setup/setup_reward_env.sh\n' >&2
            printf 'For an existing custom environment, set LIGHTX2V_REWARD_PYTHON to its executable.\n' >&2
        fi
        exit 1
    fi
done
command -v setsid >/dev/null || { printf 'The setsid utility is required.\n' >&2; exit 1; }

# Refuse occupied ports: a healthy unrelated process must never satisfy readiness.
"$RL_UR_PYTHON" - <<'PY'
import os
import socket
import torch

host = os.environ['LIGHTX2V_REWARD_HOST']
if host in {'localhost', '0.0.0.0', '::', '::1'} or host.startswith('127.') or '://' in host:
    raise SystemExit('LIGHTX2V_REWARD_HOST must be a node-routable hostname or IP, without a URL scheme.')
socket.getaddrinfo(host, None)
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit('The reward pool requires exactly eight visible CUDA GPUs.')
ports = [8791, 8792, 8793, 8794, 8795, 8796, 8797, 8798]
sockets = []
try:
    for port in ports:
        sock = socket.socket()
        sockets.append(sock)
        sock.bind(('0.0.0.0', port))
finally:
    for sock in sockets:
        sock.close()
PY

RL_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RL_LOG_DIR="$RL_MODULE_DIR/outputs/reward_pool/$RL_RUN_ID"
mkdir -p "$RL_LOG_DIR"
printf '%s\n' "$$" > "$RL_LOG_DIR/supervisor.pid"
RL_CHILD_PIDS=()
RL_CHILD_NAMES=()
RL_HEALTH_URLS=()
cleanup() {
    local status=$? pid deadline alive
    trap - EXIT INT TERM
    # setsid gives every child its own process group, including vLLM descendants.
    for pid in "${RL_CHILD_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    deadline=$((SECONDS + 10))
    while (( SECONDS < deadline )); do
        alive=0
        for pid in "${RL_CHILD_PIDS[@]}"; do
            if kill -0 -- "-$pid" 2>/dev/null; then alive=1; fi
        done
        if (( ! alive )); then break; fi
        sleep 1
    done
    for pid in "${RL_CHILD_PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
    for pid in "${RL_CHILD_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    printf 'Reward pool stopped. Logs: %s\n' "$RL_LOG_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for RL_GPU in 0 1 2 3 4 5; do
    RL_PORT=$((8792 + RL_GPU))
    RL_NAME="unifiedreward-gpu${RL_GPU}-port${RL_PORT}"
    setsid env CUDA_VISIBLE_DEVICES="$RL_GPU" \
        LIGHTX2V_UNIFIEDREWARD_HOST=0.0.0.0 LIGHTX2V_UNIFIEDREWARD_PORT="$RL_PORT" \
        LIGHTX2V_UNIFIEDREWARD_TP=1 LIGHTX2V_UNIFIEDREWARD_PYTHON="$RL_UR_PYTHON" \
        bash "$RL_SCRIPT_DIR/start_unifiedreward_server.sh" > "$RL_LOG_DIR/$RL_NAME.log" 2>&1 &
    RL_CHILD_PIDS+=("$!")
    RL_CHILD_NAMES+=("$RL_NAME")
    RL_HEALTH_URLS+=("http://127.0.0.1:$RL_PORT/health")
    printf '%s\n' "$!" > "$RL_LOG_DIR/$RL_NAME.pid"
done
for RL_GPU in 6 7; do
    RL_PORT=8791
    if (( RL_GPU == 7 )); then RL_PORT=8798; fi
    RL_NAME="hpsv3-gpu${RL_GPU}-port${RL_PORT}"
    setsid env CUDA_VISIBLE_DEVICES="$RL_GPU" LIGHTX2V_REWARD_PYTHON="$RL_HPS_PYTHON" \
        bash "$RL_SCRIPT_DIR/start_reward_server.sh" --host 0.0.0.0 --port "$RL_PORT" \
        --models hpsv3 --hps-device cuda:0 > "$RL_LOG_DIR/$RL_NAME.log" 2>&1 &
    RL_CHILD_PIDS+=("$!")
    RL_CHILD_NAMES+=("$RL_NAME")
    RL_HEALTH_URLS+=("http://127.0.0.1:$RL_PORT/health")
    printf '%s\n' "$!" > "$RL_LOG_DIR/$RL_NAME.pid"
done

check_children() {
    local index status
    for index in "${!RL_CHILD_PIDS[@]}"; do
        if ! kill -0 "${RL_CHILD_PIDS[$index]}" 2>/dev/null; then
            status=0
            wait "${RL_CHILD_PIDS[$index]}" || status=$?
            printf 'Reward process %s exited (%s); see %s/%s.log\n' \
                "${RL_CHILD_NAMES[$index]}" "$status" "$RL_LOG_DIR" "${RL_CHILD_NAMES[$index]}" >&2
            tail -n 40 "$RL_LOG_DIR/${RL_CHILD_NAMES[$index]}.log" >&2 || true
            exit 1
        fi
    done
}
RL_DEADLINE=$((SECONDS + RL_STARTUP_TIMEOUT))
printf 'Waiting for eight reward servers. Logs: %s\n' "$RL_LOG_DIR"
while true; do
    check_children
    if "$RL_UR_PYTHON" - "${RL_HEALTH_URLS[@]}" <<'PY'
import concurrent.futures
import sys
import urllib.request

def ready(url):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except Exception:
        return False

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    raise SystemExit(0 if all(pool.map(ready, sys.argv[1:])) else 1)
PY
    then
        check_children
        break
    fi
    if (( SECONDS >= RL_DEADLINE )); then
        printf 'Reward readiness timed out after %s seconds.\n' "$RL_STARTUP_TIMEOUT" >&2; exit 1
    fi
    sleep 2
done
printf 'READY: UnifiedReward http://%s:8792..8797; HPSv3 http://%s:8791 and :8798\n' \
    "$LIGHTX2V_REWARD_HOST" "$LIGHTX2V_REWARD_HOST"
while true; do check_children; sleep 1; done
