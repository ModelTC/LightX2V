#!/usr/bin/env bash
# Default: Geo + the legacy HPSv3 general/percentile and VideoAlign MQ recipe.
# All CLI arguments are forwarded; a later --config/--reward-config overrides defaults.
set -euo pipefail

RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_MODULE_DIR="$(cd -- "$RL_SCRIPT_DIR/../.." && pwd)"
cd "$RL_MODULE_DIR/.."
export PYTHONPATH="$RL_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
RL_TRAIN_PYTHON="${LIGHTX2V_RL_PYTHON:-$RL_MODULE_DIR/.venv-rl/bin/python}"
if [[ ! -x "$RL_TRAIN_PYTHON" ]]; then
    printf 'Training Python not found: %s\nCreate and install lightx2v_rl/.venv-rl first.\n' "$RL_TRAIN_PYTHON" >&2
    exit 1
fi

exec "$RL_TRAIN_PYTHON" -m lightx2v_rl.train --config wan_geo --reward-config hpsv3_videoalign "$@"
