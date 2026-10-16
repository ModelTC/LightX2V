#!/usr/bin/env bash
# Start HPSv3/VideoAlign using its own interpreter, regardless of active venv.
set -euo pipefail

RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_MODULE_DIR="$(cd -- "$RL_SCRIPT_DIR/../.." && pwd)"
cd "$RL_MODULE_DIR/.."
export PYTHONPATH="$RL_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RL_MODULE_DIR/models/.hf-runtime}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RL_REWARD_PYTHON="${LIGHTX2V_REWARD_PYTHON:-$RL_MODULE_DIR/.venv-reward/bin/python}"
if [[ ! -x "$RL_REWARD_PYTHON" ]]; then
    printf 'Reward Python not found: %s\nRun: bash %s/../setup/setup_reward_env.sh\n' "$RL_REWARD_PYTHON" "$RL_SCRIPT_DIR" >&2
    exit 1
fi

exec "$RL_REWARD_PYTHON" -m lightx2v_rl.rewards.server "$@"
