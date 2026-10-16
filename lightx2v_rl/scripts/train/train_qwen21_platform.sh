#!/usr/bin/env bash
# The training platform executes this command once on every allocated GPU node.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash lightx2v_rl/scripts/train/train_qwen21_platform.sh [key=value overrides]

Submit this command to a platform that runs it once on each of four or five eight-GPU
nodes. The platform must provide the same MASTER_ADDR and MASTER_PORT on all nodes.
MASTER_PORT is the torchrun rendezvous port and must differ from Ray's port 6379.

This launcher starts one CPU bootstrap per node. The recipe selects the topology:
  *_32gpu: 4 total nodes = 3 Ray training nodes + 1 reward node.
  *_40gpu: 5 total nodes = 4 Ray training nodes + 1 reward node.
The last rank runs six UnifiedReward and two HPSv3 services. Only rank 0 starts
the training driver. The GPU suffix includes reward GPUs.

Optional environment:
  LIGHTX2V_RL_PYTHON    Training interpreter (default: lightx2v_rl/.venv-rl/bin/python)
  LIGHTX2V_RL_CONFIG    qwen_image21_turbo_geo_40gpu (default),
                       qwen_image21_turbo_{geo,cps,flow_grpo}_{32,40}gpu
  LIGHTX2V_RDZV_ID      Shared job-specific rendezvous ID (default: 5236)
  LIGHTX2V_PLATFORM_STARTUP_TIMEOUT  Service startup timeout in seconds (default: 1800)
  GLOO_SOCKET_IFNAME    CPU control-plane interface, when explicit selection is needed
  NCCL_SOCKET_IFNAME    Training communication interface, when required by the cluster
  LIGHTX2V_RAY_DASHBOARD_HOST  Head dashboard bind address (default: 127.0.0.1)
  LIGHTX2V_RAY_DASHBOARD_PORT  Head dashboard port (default: 8265)

Do not wrap this command in another eight-process torchrun invocation. All
nodes need the training interpreter; all nodes must use the same shared mount.
Service logs: lightx2v_rl/outputs/platform/<run>/rank<rank>-<pid>/
Training stdout remains visible in the platform job log and is saved to rank0's train.log.
EOF
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
: "${MASTER_ADDR:?The platform must set the same MASTER_ADDR on all nodes}"
: "${MASTER_PORT:?The platform must set the same MASTER_PORT on all nodes}"
if [[ ! "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$MASTER_PORT < 1 || 10#$MASTER_PORT > 65535 )); then
    printf 'MASTER_PORT must be an integer from 1 to 65535.\n' >&2
    exit 2
fi
if (( 10#$MASTER_PORT == 6379 )); then
    printf 'MASTER_PORT=6379 conflicts with the Ray head port; choose another rendezvous port.\n' >&2
    exit 2
fi

export LIGHTX2V_RL_CONFIG="${LIGHTX2V_RL_CONFIG:-qwen_image21_turbo_geo_40gpu}"
case "$LIGHTX2V_RL_CONFIG" in
    qwen_image21_turbo_geo_32gpu|qwen_image21_turbo_cps_32gpu|qwen_image21_turbo_flow_grpo_32gpu) RL_PLATFORM_NODES=4 ;;
    qwen_image21_turbo_geo_40gpu|qwen_image21_turbo_cps_40gpu|qwen_image21_turbo_flow_grpo_40gpu) RL_PLATFORM_NODES=5 ;;
    *) printf 'This launcher requires a qwen_image21_turbo_*_32gpu or *_40gpu recipe.\n' >&2; exit 2 ;;
esac

RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_MODULE_DIR="$(cd -- "$RL_SCRIPT_DIR/../.." && pwd)"
cd "$RL_MODULE_DIR/.."
RL_PYTHON="${LIGHTX2V_RL_PYTHON:-$RL_MODULE_DIR/.venv-rl/bin/python}"
if [[ "$RL_PYTHON" != /* ]]; then RL_PYTHON="$PWD/$RL_PYTHON"; fi
if [[ ! -x "$RL_PYTHON" ]]; then
    printf 'Training interpreter unavailable: %s\nRun setup_qwen21_env.sh on the GPU cluster first.\n' "$RL_PYTHON" >&2
    exit 1
fi
export LIGHTX2V_RL_PYTHON="$RL_PYTHON"
export PYTHONPATH="$RL_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/lightx2v_rl/models/.hf-runtime}"
if [[ "$HF_HOME" != /* ]]; then export HF_HOME="$PWD/$HF_HOME"; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Catch a replaced PyPI Diffusers before allocating Ray workers and loading the
# eight reward models. This imports definitions only, without model weights.
"$RL_PYTHON" - <<'PY'
import sys
try:
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel
except ImportError as exc:
    raise SystemExit(
        f'Qwen-Image-2.1 training environment check failed ({sys.executable}): {exc}\n'
        'The training environment requires the pinned official Diffusers commit in '
        'lightx2v_rl/requirements/qwen_image21.txt; PyPI diffusers 0.40.0 lacks QwenImage21.\n'
        'Run bash lightx2v_rl/scripts/setup/repair_qwen21_diffusers.sh once on the GPU host, then resubmit.'
    ) from None
PY

exec "$RL_PYTHON" -m torch.distributed.run \
    --nnodes="$RL_PLATFORM_NODES" --nproc-per-node=1 --max-restarts=0 \
    --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="${LIGHTX2V_RDZV_ID:-5236}" \
    --module lightx2v_rl.deployment.qwen_image.platform "$@"
