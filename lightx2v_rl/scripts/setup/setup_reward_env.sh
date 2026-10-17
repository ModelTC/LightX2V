#!/usr/bin/env bash
# Install the legacy reward stack without touching .venv-rl.
set -euo pipefail

RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_MODULE_DIR="$(cd -- "$RL_SCRIPT_DIR/../.." && pwd)"
RL_REWARD_ENV="$RL_MODULE_DIR/.venv-reward"
RL_REWARD_PYTHON="$RL_REWARD_ENV/bin/python"
cd "$RL_MODULE_DIR"

if [[ ! -x "$RL_REWARD_PYTHON" ]]; then
    if [[ -e "$RL_REWARD_ENV" || -L "$RL_REWARD_ENV" ]]; then
        printf 'Existing reward environment has an unavailable interpreter: %s\nRun setup on the GPU host; this script will not overwrite it.\n' "$RL_REWARD_ENV" >&2
        exit 1
    fi
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "${REWARD_PYTHON_VERSION:-3.12}" --seed "$RL_REWARD_ENV"
    else
        if ! "${REWARD_BOOTSTRAP_PYTHON:-python3}" -c 'import ensurepip' >/dev/null 2>&1; then
            printf 'Python ensurepip is unavailable. Install uv and rerun setup, or install the matching python3-venv package (python3.12-venv for Python 3.12).\n' >&2
            exit 1
        fi
        "${REWARD_BOOTSTRAP_PYTHON:-python3}" -m venv "$RL_REWARD_ENV"
    fi
fi

"$RL_REWARD_PYTHON" - "$RL_REWARD_ENV" "$RL_MODULE_DIR/.venv-rl" <<'PY_ENV'
import sys
from pathlib import Path
expected = Path(sys.argv[1]).resolve()
if (not (expected / 'pyvenv.cfg').is_file() or Path(sys.prefix).resolve() != expected
        or sys.prefix == sys.base_prefix or expected == Path(sys.argv[2]).resolve()):
    raise SystemExit('Refusing to install HPSv3 outside its isolated reward virtual environment.')
PY_ENV

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$RL_REWARD_PYTHON" --upgrade pip
    reward_pip() { uv pip install --python "$RL_REWARD_PYTHON" "$@"; }
else
    if ! "$RL_REWARD_PYTHON" -m pip --version >/dev/null 2>&1; then
        if ! "$RL_REWARD_PYTHON" -m ensurepip --upgrade; then
            printf 'Reward environment exists but pip is missing. Install uv and rerun setup; the existing environment can be repaired without deleting it.\n' >&2
            exit 1
        fi
    fi
    reward_pip() { "$RL_REWARD_PYTHON" -m pip install "$@"; }
fi

reward_pip torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
reward_pip -r "$RL_MODULE_DIR/requirements/reward.txt"
reward_pip --no-deps -e "$RL_MODULE_DIR"

"$RL_REWARD_PYTHON" - <<'PY'
import sys
import peft
import torch
import transformers
print("Reward Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Transformers:", transformers.__version__, "PEFT:", peft.__version__)
assert transformers.__version__ == "4.51.3"
assert peft.__version__ == "0.17.0"
PY

printf '\nReward environment ready: %s\n' "$RL_REWARD_ENV"
printf 'Start the service: bash %q/../rewards/start_reward_server.sh --help\n' "$RL_SCRIPT_DIR"
