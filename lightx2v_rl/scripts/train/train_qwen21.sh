#!/usr/bin/env bash
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RL_SCRIPT_DIR/../../.."
RL_PYTHON="${LIGHTX2V_RL_PYTHON:-lightx2v_rl/.venv-rl/bin/python}"
if [[ ! -x "$RL_PYTHON" ]]; then
    printf 'Training Python unavailable; run setup_qwen21_env.sh on the GPU host.\n' >&2
    exit 1
fi
export PYTHONPATH="$PWD/lightx2v_rl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/lightx2v_rl/models/.hf-runtime}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec "$RL_PYTHON" -m lightx2v_rl.train --config "${LIGHTX2V_RL_CONFIG:-qwen_image21_turbo_geo}" "$@"
