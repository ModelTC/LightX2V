#!/usr/bin/env bash
# Official UR-Edit vLLM endpoint; CUDA_VISIBLE_DEVICES selects its GPUs.
set -euo pipefail
RL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RL_MODULE_DIR="$(cd -- "$RL_SCRIPT_DIR/../.." && pwd)"
cd "$RL_MODULE_DIR/.."
RL_UR_PYTHON="${LIGHTX2V_UNIFIEDREWARD_PYTHON:-lightx2v_rl/.venv-unifiedreward/bin/python}"
RL_UR_MODEL="${LIGHTX2V_UNIFIEDREWARD_MODEL:-lightx2v_rl/models/rewards/unifiedreward}"
if [[ ! -x "$RL_UR_PYTHON" ]]; then
    printf 'UnifiedReward Python missing; run bash lightx2v_rl/scripts/setup/setup_unifiedreward_env.sh\n' >&2
    exit 1
fi
"$RL_UR_PYTHON" - "$RL_UR_MODEL" <<'PY'
import json
import sys
import sysconfig
from pathlib import Path

# Triton compiles a Python extension while vLLM inspects Qwen3.5. These headers
# belong to the node's base Python, even when the virtualenv is on shared storage.
header = Path(sysconfig.get_path('include')) / 'Python.h'
if not header.is_file():
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    raise SystemExit(
        f'UnifiedReward requires Python development headers for Triton; missing: {header}\n'
        f'For system Python on Debian/Ubuntu, run on each GPU node (or in its image):\n'
        f'  apt-get update && apt-get install -y python{version}-dev\n'
        'A shared virtualenv does not provide the system headers. Model downloads and pip reinstall are unnecessary.'
    )

root = Path(sys.argv[1])
if not (root / 'download-manifest.json').is_file():
    raise SystemExit(
        f'UnifiedReward snapshot missing at {root}; finish download_qwen21_assets.py and run '
        'lightx2v_rl/scripts/assets/link_resources.py and link_reward_weights.py '
        '(or set LIGHTX2V_UNIFIEDREWARD_MODEL).'
    )
manifest = json.loads((root / 'download-manifest.json').read_text())
if not manifest.get('complete'):
    raise SystemExit('UnifiedReward snapshot is incomplete; finish download_qwen21_assets.py first')
for item in manifest['files']:
    path = root / item['rfilename']
    if not path.is_file() or path.stat().st_size != item['size']:
        raise SystemExit(f'Missing or truncated model asset: {path}')
PY
export HF_HOME="${HF_HOME:-lightx2v_rl/models/.hf-runtime}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# vLLM 0.20.2 selects this through CLI; the old GDN environment flag is ignored.
exec "$RL_UR_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$RL_UR_MODEL" --served-model-name UnifiedReward \
    --host "${LIGHTX2V_UNIFIEDREWARD_HOST:-127.0.0.1}" \
    --port "${LIGHTX2V_UNIFIEDREWARD_PORT:-8792}" \
    --tensor-parallel-size "${LIGHTX2V_UNIFIEDREWARD_TP:-1}" \
    --gpu-memory-utilization "${LIGHTX2V_UNIFIEDREWARD_GPU_MEMORY:-0.85}" \
    --max-model-len "${LIGHTX2V_UNIFIEDREWARD_MAX_MODEL_LEN:-65536}" \
    --gdn-prefill-backend triton \
    --limit-mm-per-prompt '{"image": 2}' \
    --mm-encoder-tp-mode data --mm-processor-cache-type shm \
    --enable-prefix-caching --default-chat-template-kwargs '{"enable_thinking": false}' "$@"
