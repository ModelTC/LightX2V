#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash scripts/minimax_h3/ref2av_smoke/start_server_sp8.sh' \
        'Run cache_adaln.sh first with the same model/config and user.' \
        'Environment: MODEL_PATH, CONFIG_JSON, PYTHON_BIN, CUDA_VISIBLE_DEVICES (8 GPUs).' \
        'HOST=127.0.0.1, PORT=8000, MAX_QUEUE_SIZE=10, optional METRIC_PORT.' \
        'Runs one ref2av service with eight Ulysses sequence-parallel workers.'
    exit 0
fi
if (( $# != 0 )); then
    printf 'Unexpected argument: %s (use --help)\n' "$1" >&2
    exit 2
fi

ref2av_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
lightx2v_path="$ref2av_repo_root"
model_path="${MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
ref2av_config="${CONFIG_JSON:-${ref2av_repo_root}/configs/minimax_h3/minimax_h3_ref2av_sp8_768p.json}"
ref2av_python="${PYTHON_BIN:-python3}"
ref2av_host="${HOST:-127.0.0.1}"
ref2av_port="${PORT:-8000}"
ref2av_queue_size="${MAX_QUEUE_SIZE:-10}"

[[ -d "$model_path/transformer_ref" ]] || { printf 'Missing reference weights directory: %s/transformer_ref\n' "$model_path" >&2; exit 1; }
[[ -f "$ref2av_config" ]] || { printf 'Missing inference config: %s\n' "$ref2av_config" >&2; exit 1; }
command -v "$ref2av_python" >/dev/null 2>&1 || { printf 'Python executable not found: %s\n' "$ref2av_python" >&2; exit 1; }
model_path="$(cd -- "$model_path" && pwd -P)"
ref2av_config="$(cd -- "$(dirname -- "$ref2av_config")" && pwd -P)/$(basename -- "$ref2av_config")"
ref2av_python="$(command -v "$ref2av_python")"
if [[ "$ref2av_python" == */* && "$ref2av_python" != /* ]]; then
    ref2av_python="$(cd -- "$(dirname -- "$ref2av_python")" && pwd -P)/$(basename -- "$ref2av_python")"
fi
[[ "$ref2av_port" =~ ^[0-9]+$ ]] && (( 10#$ref2av_port >= 1 && 10#$ref2av_port <= 65535 )) || { printf 'PORT must be between 1 and 65535.\n' >&2; exit 2; }
[[ "$ref2av_queue_size" =~ ^[1-9][0-9]*$ ]] || { printf 'MAX_QUEUE_SIZE must be a positive integer.\n' >&2; exit 2; }

export PLATFORM="${PLATFORM:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a ref2av_gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#ref2av_gpus[@]} != 8 )) || [[ "$CUDA_VISIBLE_DEVICES" == *, || "$CUDA_VISIBLE_DEVICES" == ,* || "$CUDA_VISIBLE_DEVICES" == *,,* ]]; then
    printf 'CUDA_VISIBLE_DEVICES must list exactly 8 GPUs for seq_p_size=8.\n' >&2
    exit 2
fi
declare -A ref2av_seen_gpus=()
for ref2av_gpu in "${ref2av_gpus[@]}"; do
    [[ -n "${ref2av_gpu//[[:space:]]/}" && "$ref2av_gpu" != "-1" ]] || { printf 'Invalid CUDA_VISIBLE_DEVICES entry.\n' >&2; exit 2; }
    [[ -z "${ref2av_seen_gpus[$ref2av_gpu]:-}" ]] || { printf 'CUDA_VISIBLE_DEVICES must list 8 distinct GPUs.\n' >&2; exit 2; }
    ref2av_seen_gpus[$ref2av_gpu]=1
done
export PYTHONPATH="${PYTHONPATH:-}"
export DTYPE="${DTYPE:-BF16}"
export SENSITIVE_LAYER_DTYPE="${SENSITIVE_LAYER_DTYPE:-BF16}"
ref2av_profiling_debug_level="${PROFILING_DEBUG_LEVEL:-0}"
source "$ref2av_repo_root/scripts/base/base.sh"
export PROFILING_DEBUG_LEVEL="$ref2av_profiling_debug_level"
cd -- "$ref2av_repo_root"

# torch.distributed.run uses the same interpreter/environment as the service.
ref2av_command=(
    "$ref2av_python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8
    -m lightx2v.server
    --model_cls minimax_h3 --model-variant ref2av
    --model_path "$model_path" --config_json "$ref2av_config"
    --host "$ref2av_host" --port "$ref2av_port" --max_queue_size "$ref2av_queue_size"
)
if [[ -n "${METRIC_PORT:-}" ]]; then
    [[ "$METRIC_PORT" =~ ^[0-9]+$ ]] && (( 10#$METRIC_PORT >= 1 && 10#$METRIC_PORT <= 65535 )) || { printf 'METRIC_PORT must be between 1 and 65535.\n' >&2; exit 2; }
    ref2av_command+=(--metric_port "$METRIC_PORT")
fi
printf 'Starting ref2av SP8 service at http://%s:%s using %s\n' "$ref2av_host" "$ref2av_port" "$ref2av_config"
exec "${ref2av_command[@]}"
