#!/usr/bin/env bash
# Run on the GPU host; dependency overrides leave upstream source unchanged.
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RL_SCRIPT_DIR/../../.."
RL_PYTHON="${LIGHTX2V_RL_PYTHON:-lightx2v_rl/.venv-rl/bin/python}"
if ! command -v uv >/dev/null 2>&1; then
    printf 'Install uv first (python3 -m pip install uv), then rerun this script.\n' >&2
    exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-lightx2v_rl/models/.uv-cache}"
export HF_HOME="${HF_HOME:-lightx2v_rl/models/.hf-runtime}"
if [[ ! -x "$RL_PYTHON" ]]; then
    if [[ -d lightx2v_rl/.venv-rl ]]; then
        printf 'Existing .venv-rl has an unavailable interpreter on this host. Run on the GPU host or set LIGHTX2V_RL_PYTHON to a working Python >=3.11 environment.\n' >&2
        exit 1
    fi
    uv venv --python 3.12 --seed lightx2v_rl/.venv-rl
fi
uv pip install --python "$RL_PYTHON" --torch-backend=auto \
    --reinstall-package diffusers \
    --override lightx2v_rl/requirements/qwen_image21.txt \
    -e 'third_party/verl-omni[gpu]' -e lightx2v_rl
"$RL_PYTHON" lightx2v_rl/scripts/setup/prepare_verl_omni.py
"$RL_PYTHON" - <<'PY'
import sys
import torch
import transformers
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
print('Python:', sys.executable)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
print('Official pipeline:', QwenImage21Pipeline.__module__)
assert transformers.__version__ == '5.14.1'
assert torch.cuda.is_available(), 'Run this setup on the GPU host before starting training.'
PY
