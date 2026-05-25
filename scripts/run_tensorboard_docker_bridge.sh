#!/usr/bin/env bash
# Docker TensorBoard bridge: container TB -> host port -> (optional) IDE port forward.
# Docs: docs/ZH_CN/source/method_tutorials/torch_profiling.md
#
# Use this ONLY when inference runs inside Docker and TensorBoard must run there too.
# For same-environment (no Docker), use the regular tensorboard CLI instead.
#
# Full example:
#   TENSORBOARD_CONTAINER=my_container \
#   LIGHTX2V_TORCH_PROFILE_TB_DIR=/path/to/LightX2V/save_results/torch_profile \
#   TENSORBOARD_PORT=16006 \
#   bash scripts/run_tensorboard_docker_bridge.sh
#
# Short example (set container name once):
#   export TENSORBOARD_CONTAINER=my_container
#   bash scripts/run_tensorboard_docker_bridge.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOGDIR="${LIGHTX2V_TORCH_PROFILE_TB_DIR:-${REPO_ROOT}/save_results/torch_profile}"
if [[ "${LOGDIR}" != /* ]]; then
  LOGDIR="${REPO_ROOT}/${LOGDIR}"
fi
PORT="${TENSORBOARD_PORT:-16006}"
CONTAINER="${TENSORBOARD_CONTAINER:-}"
PROXY_SCRIPT="/tmp/lightx2v_tb_proxy_${PORT}.py"
PROXY_LOG="/tmp/lightx2v_tb_proxy_${PORT}.log"

usage() {
  cat <<EOF
Usage: bash scripts/run_tensorboard_docker_bridge.sh

Required:
  TENSORBOARD_CONTAINER       Docker container name or ID

Optional:
  LIGHTX2V_TORCH_PROFILE_TB_DIR  Logdir (default: ${REPO_ROOT}/save_results/torch_profile)
  TENSORBOARD_PORT            Host listen port (default: 16006)

After startup, if you use Remote SSH, forward port \${TENSORBOARD_PORT} in your IDE,
then open: http://127.0.0.1:\${TENSORBOARD_PORT}/#pytorch_profiler
EOF
}

ensure_torch_tb_profiler_in_container() {
  local container="$1"
  if ! docker exec "${container}" bash -lc "python3 -c 'import torch_tb_profiler'" 2>/dev/null; then
    echo "Installing torch-tb-profiler in container ${container}..."
    docker exec "${container}" bash -lc "pip install torch-tb-profiler -q"
  fi
}

stop_port_listener() {
  if command -v ss >/dev/null 2>&1 && ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "Stopping process listening on host port ${PORT}..."
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi
    sleep 1
  fi
}

write_tcp_proxy() {
  local target_host="$1"
  cat > "${PROXY_SCRIPT}" <<EOF
#!/usr/bin/env python3
import socket, threading
LISTEN = ("0.0.0.0", ${PORT})
TARGET = ("${target_host}", ${PORT})
BUF = 65536

def pipe(a, b):
    try:
        while True:
            data = a.recv(BUF)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

def handle(client):
    try:
        upstream = socket.create_connection(TARGET, timeout=10)
    except OSError as e:
        print(f"connect failed: {e}", flush=True)
        client.close()
        return
    threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(LISTEN)
srv.listen(128)
print(f"proxy {LISTEN} -> {TARGET}", flush=True)
while True:
    client, _ = srv.accept()
    threading.Thread(target=handle, args=(client,), daemon=True).start()
EOF
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  if [[ -z "${CONTAINER}" ]]; then
    echo "TENSORBOARD_CONTAINER is required." >&2
    usage
    exit 1
  fi
  if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    echo "Docker container not found: ${CONTAINER}" >&2
    exit 1
  fi

  ensure_torch_tb_profiler_in_container "${CONTAINER}"
  docker exec "${CONTAINER}" bash -lc "pkill -f 'tensorboard.*--port ${PORT}'" 2>/dev/null || true
  sleep 1

  echo "Starting TensorBoard in container ${CONTAINER} (port ${PORT})..."
  echo "Logdir: ${LOGDIR}"
  docker exec -d "${CONTAINER}" bash -lc \
    "tensorboard --logdir '${LOGDIR}' --port ${PORT} --bind_all"
  sleep 2

  local ip
  ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${CONTAINER}")"
  if [[ -z "${ip}" ]]; then
    echo "Failed to resolve container IP for ${CONTAINER}" >&2
    exit 1
  fi

  stop_port_listener
  write_tcp_proxy "${ip}"
  nohup python3 "${PROXY_SCRIPT}" > "${PROXY_LOG}" 2>&1 &
  sleep 1

  if curl -sf "http://127.0.0.1:${PORT}/" >/dev/null; then
    echo "TensorBoard ready: http://127.0.0.1:${PORT}/#pytorch_profiler"
    echo "Open the PYTORCH PROFILER tab (not SCALARS)."
    echo "Remote SSH: forward port ${PORT} in IDE, then open the URL above."
  else
    echo "Bridge failed. Check ${PROXY_LOG}" >&2
    exit 1
  fi
}

main "$@"
