#!/usr/bin/env bash
# Repair only Diffusers in the GPU training venv; prefer the prepared offline wheel.
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RL_SCRIPT_DIR/../../.."
RL_PYTHON="${LIGHTX2V_RL_PYTHON:-lightx2v_rl/.venv-rl/bin/python}"
if [[ ! -x "$RL_PYTHON" ]]; then
    printf 'Training Python unavailable: %s\nRun this script on the GPU host with the shared repository mounted.\n' "$RL_PYTHON" >&2
    exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/lightx2v_rl/models/.uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/lightx2v_rl/models/.pip-cache}"
export HF_HOME="${HF_HOME:-$PWD/lightx2v_rl/models/.hf-runtime}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RL_DIFFUSERS_REV=62cde2b0115d57492d2f5ecefab54994797df2e3
RL_DIFFUSERS_WHEEL="lightx2v_rl/models/.runtime-wheels/diffusers-$RL_DIFFUSERS_REV/diffusers-0.41.0.dev0-py3-none-any.whl"

"$RL_PYTHON" - <<'PY'
import sys
from pathlib import Path
prefix = Path(sys.prefix).resolve()
reward_envs = {Path('lightx2v_rl', name).resolve() for name in ('.venv-reward', '.venv-unifiedreward')}
if sys.prefix == sys.base_prefix or not (prefix / 'pyvenv.cfg').is_file() or prefix in reward_envs:
    raise SystemExit('Refusing to repair Diffusers outside an isolated training virtual environment.')
PY

if [[ -f "$RL_DIFFUSERS_WHEEL" ]]; then
    "$RL_PYTHON" - "$RL_DIFFUSERS_WHEEL" "$RL_DIFFUSERS_REV" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
wheel = Path(sys.argv[1])
manifest = json.loads(wheel.with_name('provenance.json').read_text())
if (manifest['revision'] != sys.argv[2] or manifest['wheel'] != wheel.name
        or manifest['sha256'] != hashlib.sha256(wheel.read_bytes()).hexdigest()):
    raise SystemExit('Offline Diffusers wheel does not match its pinned source manifest.')
PY
    RL_DIFFUSERS_SPEC="$RL_DIFFUSERS_WHEEL"
else
    RL_DIFFUSERS_SPEC="diffusers @ git+https://github.com/huggingface/diffusers.git@$RL_DIFFUSERS_REV"
fi
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$RL_PYTHON" --no-deps --reinstall-package diffusers "$RL_DIFFUSERS_SPEC"
else
    "$RL_PYTHON" -m pip install --no-deps --force-reinstall "$RL_DIFFUSERS_SPEC"
fi
"$RL_PYTHON" - <<'PY'
import sys
import diffusers
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel
print('Training Python:', sys.executable)
print('Diffusers:', diffusers.__version__, diffusers.__file__)
print('QwenImage21 imports OK:', QwenImage21Pipeline.__name__, QwenImage21Transformer2DModel.__name__,
      AutoencoderKLQwenImage21.__name__)
PY
