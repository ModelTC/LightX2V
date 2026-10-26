#!/usr/bin/env bash
# Three or four eight-GPU Ray nodes, plus one separate reward node.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash lightx2v_rl/scripts/train/qwen21_cluster.sh head|worker|train [training overrides]

Required on every training node:
  LIGHTX2V_HEAD_IP       Routable IP of the Ray head node.
  LIGHTX2V_NODE_IP       This node's routable IP (equals HEAD_IP on the head).
  LIGHTX2V_REWARD_HOST   Routable address of the separate reward node.

head:   Start the eight-GPU head on port 6379 and remain in the foreground.
worker: Join the head with eight GPUs and remain in the foreground.
train:  Run on the head, after all workers and all eight reward servers are ready.
        Require 3 Ray nodes for *_32gpu or 4 for *_40gpu recipes.
        Never start a local replacement cluster.

Use the identical shared repository/model mount path and environment on all nodes.
Set LIGHTX2V_RL_CONFIG to qwen_image21_turbo_{geo,cps,flow_grpo}_{32,40}gpu.
Default: qwen_image21_turbo_geo_40gpu; the suffix includes the reward GPUs.
Optional on the head: LIGHTX2V_RAY_DASHBOARD_HOST (default 127.0.0.1),
LIGHTX2V_RAY_DASHBOARD_PORT (default 8265). Set host=0.0.0.0 for a trusted
internal platform proxy; these settings apply only when starting a new head.
No command stops an unrelated Ray cluster or other users' processes.
EOF
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
RL_ACTION="${1:-}"
case "$RL_ACTION" in head|worker|train) shift ;; *) usage >&2; exit 2 ;; esac
if [[ "$RL_ACTION" != train ]] && (( $# )); then usage >&2; exit 2; fi
: "${LIGHTX2V_HEAD_IP:?Set LIGHTX2V_HEAD_IP to the Ray head node IP}"
: "${LIGHTX2V_NODE_IP:?Set LIGHTX2V_NODE_IP to this training node IP}"
: "${LIGHTX2V_REWARD_HOST:?Set LIGHTX2V_REWARD_HOST to the dedicated reward node address}"

RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_MODULE_DIR="$(cd -- "$RL_SCRIPT_DIR/../.." && pwd)"
cd "$RL_MODULE_DIR/.."
RL_REPOSITORY_ROOT="$PWD"
RL_PYTHON="${LIGHTX2V_RL_PYTHON:-$RL_MODULE_DIR/.venv-rl/bin/python}"
if [[ ! -x "$RL_PYTHON" ]]; then
    printf 'Training interpreter unavailable: %s\n' "$RL_PYTHON" >&2; exit 1
fi
export LIGHTX2V_RL_PYTHON="$RL_PYTHON"
export PYTHONPATH="$RL_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RL_REPOSITORY_ROOT/lightx2v_rl/models/.hf-runtime}"
if [[ "$HF_HOME" != /* ]]; then export HF_HOME="$RL_REPOSITORY_ROOT/$HF_HOME"; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LIGHTX2V_RL_CONFIG="${LIGHTX2V_RL_CONFIG:-qwen_image21_turbo_geo_40gpu}"
export RAY_ADDRESS="$LIGHTX2V_HEAD_IP:6379"
case "$LIGHTX2V_RL_CONFIG" in
    qwen_image21_turbo_geo_32gpu|qwen_image21_turbo_cps_32gpu|qwen_image21_turbo_flow_grpo_32gpu) LIGHTX2V_TRAIN_NODES=3 ;;
    qwen_image21_turbo_geo_40gpu|qwen_image21_turbo_cps_40gpu|qwen_image21_turbo_flow_grpo_40gpu) LIGHTX2V_TRAIN_NODES=4 ;;
    *) printf 'This launcher requires a qwen_image21_turbo_*_32gpu or *_40gpu recipe.\n' >&2; exit 2 ;;
esac
export LIGHTX2V_TRAIN_NODES
if [[ "$RL_ACTION" == train ]]; then
    for RL_OVERRIDE in "$@"; do
        RL_OVERRIDE_KEY="${RL_OVERRIDE%%=*}"
        RL_OVERRIDE_KEY="${RL_OVERRIDE_KEY#+}"
        RL_OVERRIDE_KEY="${RL_OVERRIDE_KEY#+}"
        case "$RL_OVERRIDE_KEY" in
            trainer.nnodes)
                if [[ "${RL_OVERRIDE#*=}" != "$LIGHTX2V_TRAIN_NODES" ]]; then
                    printf 'This cluster launcher requires trainer.nnodes=%s.\n' "$LIGHTX2V_TRAIN_NODES" >&2; exit 2
                fi ;;
            trainer.n_gpus_per_node)
                if [[ "${RL_OVERRIDE#*=}" != 8 ]]; then
                    printf 'This cluster launcher requires trainer.n_gpus_per_node=8.\n' >&2; exit 2
                fi ;;
        esac
    done
fi

"$RL_PYTHON" - "$RL_ACTION" <<'PY'
import ipaddress
import os
import sys

for key in ('LIGHTX2V_HEAD_IP', 'LIGHTX2V_NODE_IP'):
    try:
        address = ipaddress.IPv4Address(os.environ[key])
    except ValueError:
        raise SystemExit(f'{key} must be an IPv4 address.') from None
    if address.is_loopback or address.is_unspecified:
        raise SystemExit(f'{key} must be routable between nodes.')
reward = os.environ['LIGHTX2V_REWARD_HOST']
if reward in {'localhost', '0.0.0.0', '::', '::1'} or reward.startswith('127.') or '://' in reward:
    raise SystemExit('LIGHTX2V_REWARD_HOST must be a routable hostname or IP, without a URL scheme.')
is_head = os.environ['LIGHTX2V_NODE_IP'] == os.environ['LIGHTX2V_HEAD_IP']
if (sys.argv[1] in {'head', 'train'}) != is_head:
    raise SystemExit('head/train require NODE_IP == HEAD_IP; worker requires a different NODE_IP.')
PY

if [[ "$RL_ACTION" == head || "$RL_ACTION" == worker ]]; then
    "$RL_PYTHON" - <<'PY'
import os
import psutil
import torch

running = []
for process in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        args = process.info['cmdline'] or []
        name = os.path.basename(args[0]) if args else process.info['name']
        if name in {'raylet', 'gcs_server'}:
            running.append(process.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
if running:
    raise SystemExit(f'Existing local Ray processes {running}; refusing to reuse or stop them.')
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit('Each training node must expose exactly eight CUDA GPUs.')
PY
    if [[ "$RL_ACTION" == head ]]; then
        RL_DASHBOARD_HOST="${LIGHTX2V_RAY_DASHBOARD_HOST:-127.0.0.1}"
        RL_DASHBOARD_PORT="${LIGHTX2V_RAY_DASHBOARD_PORT:-8265}"
        if [[ ! "$RL_DASHBOARD_PORT" =~ ^[0-9]{1,5}$ ]] || \
            (( 10#$RL_DASHBOARD_PORT < 1 || 10#$RL_DASHBOARD_PORT > 65535 || 10#$RL_DASHBOARD_PORT == 6379 )); then
            printf 'LIGHTX2V_RAY_DASHBOARD_PORT must be 1..65535 and differ from Ray GCS port 6379.\n' >&2
            exit 2
        fi
        exec "$RL_PYTHON" -m ray.scripts.scripts start --head --port=6379 \
            --dashboard-host="$RL_DASHBOARD_HOST" --dashboard-port="$RL_DASHBOARD_PORT" \
            --node-ip-address="$LIGHTX2V_NODE_IP" --num-gpus=8 --disable-usage-stats --block
    fi
    exec "$RL_PYTHON" -m ray.scripts.scripts start --address="$RAY_ADDRESS" \
        --node-ip-address="$LIGHTX2V_NODE_IP" --num-gpus=8 --disable-usage-stats --block
fi

# Explicit address prevents ray.init from creating a local fallback. Check the
# environment on each node too: driver exports cannot repair an old Ray daemon.
"$RL_PYTHON" - "$RL_REPOSITORY_ROOT" <<'PY'
import os
import socket
import sys
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

root = sys.argv[1]
ray.init(address=os.environ['RAY_ADDRESS'], logging_level='ERROR')
try:
    nodes = [node for node in ray.nodes() if node['Alive'] and node['Resources'].get('GPU', 0) > 0]
    expected_nodes = int(os.environ['LIGHTX2V_TRAIN_NODES'])
    if len(nodes) != expected_nodes or any(node['Resources'].get('GPU') != 8 for node in nodes):
        inventory = [(node['NodeManagerAddress'], node['Resources'].get('GPU')) for node in nodes]
        raise SystemExit(f'Expected exactly {expected_nodes} live eight-GPU Ray nodes '
                         f'({expected_nodes * 8} GPUs); found {inventory}.')
    if os.environ['LIGHTX2V_HEAD_IP'] not in {node['NodeManagerAddress'] for node in nodes}:
        raise SystemExit('The configured head is not one of the Ray GPU nodes.')
    reward_ips = {item[4][0] for item in socket.getaddrinfo(os.environ['LIGHTX2V_REWARD_HOST'], None)}
    if reward_ips.intersection(node['NodeManagerAddress'] for node in nodes):
        raise SystemExit('The reward node must be separate from all Ray GPU nodes.')

    @ray.remote(num_cpus=0)
    def check_node(expected_root, expected_reward_host, expected_python, expected_prefix, expected_hf_home):
        import importlib.util
        import os
        import sys
        from pathlib import Path
        problems = []
        root_path = Path(expected_root)
        for relative in ('lightx2v_rl/lightx2v_rl/train.py', 'third_party/verl-omni/verl_omni',
                         'lightx2v_rl/models/Qwen-Image-2.1-Turbo/model_index.json',
                         'lightx2v_rl/data/opengpt4o/train.jsonl'):
            if not (root_path / relative).exists():
                problems.append(f'missing shared path: {root_path / relative}')
        if Path(sys.executable).resolve() != Path(expected_python).resolve():
            problems.append(f'worker interpreter differs: {sys.executable}')
        if Path(sys.prefix).resolve() != Path(expected_prefix).resolve():
            problems.append(f'worker virtual environment differs: {sys.prefix}')
        package = importlib.util.find_spec('lightx2v_rl')
        expected_package = root_path / 'lightx2v_rl' / 'lightx2v_rl' / '__init__.py'
        if package is None or package.origin is None or Path(package.origin).resolve() != expected_package.resolve():
            problems.append('worker imports lightx2v_rl from a different checkout')
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
            if os.environ.get(key) != '1':
                problems.append(f'{key} must be exported before Ray starts')
        if os.environ.get('LIGHTX2V_REWARD_HOST') != expected_reward_host:
            problems.append('LIGHTX2V_REWARD_HOST differs or was not exported before Ray starts')
        if os.environ.get('HF_HOME') != expected_hf_home:
            problems.append('HF_HOME differs or was not exported before Ray starts')
        if str(root_path / 'lightx2v_rl') not in os.environ.get('PYTHONPATH', '').split(os.pathsep):
            problems.append('repository PYTHONPATH was not exported before Ray starts')
        return problems

    checks = [check_node.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node['NodeID'], soft=False))
              .remote(root, os.environ['LIGHTX2V_REWARD_HOST'], sys.executable, sys.prefix, os.environ['HF_HOME'])
              for node in nodes]
    results = ray.get(checks, timeout=120)
    failures = [(node['NodeManagerAddress'], problems) for node, problems in zip(nodes, results) if problems]
    if failures:
        raise SystemExit(f'Cluster shared-path/environment check failed: {failures}')
    print(f'Cluster verified: {expected_nodes} training nodes, {expected_nodes * 8} GPUs, '
          'identical shared paths and offline environment.')
finally:
    ray.shutdown()
PY

"$RL_PYTHON" -m lightx2v_rl.models.qwen_image.preflight --config "$LIGHTX2V_RL_CONFIG" "$@"
exec bash "$RL_SCRIPT_DIR/train_qwen21.sh" "$@" ray_kwargs.ray_init.address=auto
