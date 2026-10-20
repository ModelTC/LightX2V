#!/usr/bin/env bash
# Isolate vLLM's pinned Torch/Transformers from RL and HPS dependencies.
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_MODULE_DIR="$(cd -- "$RL_SCRIPT_DIR/../.." && pwd)"
RL_UR_ENV="${LIGHTX2V_UNIFIEDREWARD_ENV:-$RL_MODULE_DIR/.venv-unifiedreward}"
RL_UR_PYTHON="$RL_UR_ENV/bin/python"
RL_SETUP_PYTHON="${LIGHTX2V_SETUP_PYTHON:-python3}"
"$RL_SETUP_PYTHON" - <<'PY_CHECK'
import platform
import sys
assert sys.version_info >= (3, 10), "Python >=3.10 is required"
libc, version = platform.libc_ver()
if platform.system() != "Linux" or libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 35):
    raise SystemExit("The pinned vLLM GPU wheel requires Linux glibc >=2.35 (e.g. Ubuntu 22.04+). "
                     "Run this setup on the GPU server or in a compatible GPU container.")
PY_CHECK

if [[ ! -x "$RL_UR_PYTHON" ]]; then
    if [[ -e "$RL_UR_ENV" || -L "$RL_UR_ENV" ]]; then
        printf 'Existing reward environment has an unavailable interpreter: %s\nRun setup on the GPU host; this script will not overwrite it.\n' "$RL_UR_ENV" >&2
        exit 1
    fi
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$RL_SETUP_PYTHON" --seed "$RL_UR_ENV"
    else
        if ! "$RL_SETUP_PYTHON" -c 'import ensurepip' >/dev/null 2>&1; then
            printf 'Python ensurepip is unavailable. Install uv and rerun setup, or install the matching python3-venv package (python3.12-venv for Python 3.12).\n' >&2
            exit 1
        fi
        "$RL_SETUP_PYTHON" -m venv "$RL_UR_ENV"
    fi
fi

# A valid venv may have been left behind by a failed stdlib ensurepip step.
# Validate its prefix before installing anything, including when an override is used.
"$RL_UR_PYTHON" - "$RL_UR_ENV" "$RL_MODULE_DIR/.venv-rl" <<'PY_ENV'
import sys
from pathlib import Path
expected = Path(sys.argv[1]).resolve()
if (not (expected / 'pyvenv.cfg').is_file() or Path(sys.prefix).resolve() != expected
        or sys.prefix == sys.base_prefix or expected == Path(sys.argv[2]).resolve()):
    raise SystemExit('Refusing to install UnifiedReward outside its isolated reward virtual environment.')
PY_ENV

if command -v uv >/dev/null 2>&1; then
    # uv can seed a pip-less venv without importing the system's ensurepip.
    uv pip install --python "$RL_UR_PYTHON" --upgrade pip
    uv pip install --python "$RL_UR_PYTHON" -r "$RL_MODULE_DIR/requirements/unifiedreward.txt"
else
    if ! "$RL_UR_PYTHON" -m pip --version >/dev/null 2>&1; then
        if ! "$RL_UR_PYTHON" -m ensurepip --upgrade; then
            printf 'Reward environment exists but pip is missing. Install uv and rerun setup; the existing environment can be repaired without deleting it.\n' >&2
            exit 1
        fi
    fi
    "$RL_UR_PYTHON" -m pip install --upgrade pip
    "$RL_UR_PYTHON" -m pip install -r "$RL_MODULE_DIR/requirements/unifiedreward.txt"
fi
