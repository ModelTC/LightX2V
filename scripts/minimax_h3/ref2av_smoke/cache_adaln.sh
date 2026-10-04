#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash scripts/minimax_h3/ref2av_smoke/cache_adaln.sh' \
        'Environment: MODEL_PATH, CONFIG_JSON, PYTHON_BIN, CUDA_VISIBLE_DEVICES (default: 0).' \
        'Builds the ref2av AdaLN cache on the first visible GPU before starting the server.' \
        'Use the same model, config, user, and cache directory as the server.' \
        'Reuses a validated matching cache; refuses to overwrite an invalid existing cache.'
    exit 0
fi
if (( $# != 0 )); then
    printf 'Unexpected argument: %s (use --help)\n' "$1" >&2
    exit 2
fi

# Resolve from this file, including when deployed under /mnt/lm_data_afs/.../codes/news.
ref2av_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
lightx2v_path="$ref2av_repo_root"
model_path="${MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
ref2av_config="${CONFIG_JSON:-${ref2av_repo_root}/configs/minimax_h3/minimax_h3_ref2av_sp8_768p.json}"
ref2av_python="${PYTHON_BIN:-python3}"

[[ -d "$model_path/transformer_ref" ]] || { printf 'Missing reference weights directory: %s/transformer_ref\n' "$model_path" >&2; exit 1; }
[[ -f "$ref2av_config" ]] || { printf 'Missing inference config: %s\n' "$ref2av_config" >&2; exit 1; }
command -v "$ref2av_python" >/dev/null 2>&1 || { printf 'Python executable not found: %s\n' "$ref2av_python" >&2; exit 1; }
model_path="$(cd -- "$model_path" && pwd -P)"
ref2av_config="$(cd -- "$(dirname -- "$ref2av_config")" && pwd -P)/$(basename -- "$ref2av_config")"
ref2av_python="$(command -v "$ref2av_python")"
if [[ "$ref2av_python" == */* && "$ref2av_python" != /* ]]; then
    ref2av_python="$(cd -- "$(dirname -- "$ref2av_python")" && pwd -P)/$(basename -- "$ref2av_python")"
fi

export PLATFORM="${PLATFORM:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
[[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != "-1" ]] || { printf 'Cache generation requires a visible GPU.\n' >&2; exit 1; }
export PYTHONPATH="${PYTHONPATH:-}"
export DTYPE="${DTYPE:-BF16}"
export SENSITIVE_LAYER_DTYPE="${SENSITIVE_LAYER_DTYPE:-BF16}"
ref2av_profiling_debug_level="${PROFILING_DEBUG_LEVEL:-0}"
source "$ref2av_repo_root/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL="$ref2av_profiling_debug_level"
cd -- "$ref2av_repo_root"

# The shared JSON supplies 29 steps and video/audio flow shifts 12/3.
# --model-variant ref2av uses transformer_ref and builds BOTH ref2av_video and
# ref2av_video_audio profiles, including the reference timesteps. This version
# uses max(video_timestep, 0.999) for visual references and 1.0 for reference
# audio; there is no separate ref_t0 option to add to the cache command.
exec "$ref2av_python" - "$model_path" "$ref2av_config" "$ref2av_repo_root" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[3]) / "tools/cache_minimax_h3_adaln"))
from builder import build_persistent_adaln_cache
from lightx2v.models.networks.minimax_h3.adaln_cache import _build_spec, _cache_path, _validate_cache
from lightx2v.utils.set_config import build_startup_config

config = build_startup_config({
    "model_cls": "minimax_h3", "task": None, "model_variant": "ref2av",
    "model_path": sys.argv[1], "config_json": sys.argv[2],
})
cache_path = _cache_path(config)
if cache_path.exists():
    if not _validate_cache(cache_path, _build_spec(config)):
        raise SystemExit(
            f"Existing AdaLN cache is invalid or incompatible: {cache_path}\n"
            "Move that directory aside manually, then rerun this script. Nothing was overwritten."
        )
    print(f"Reusing validated MiniMax-H3 ref2av AdaLN cache: {cache_path}")
else:
    print(f"Building MiniMax-H3 ref2av AdaLN cache: {cache_path}")
    build_persistent_adaln_cache(config)
PY
