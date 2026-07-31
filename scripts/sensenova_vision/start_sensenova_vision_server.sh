#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
model_path="${model_path:-/data/nvme0/lhd_codes/SenseNova-Vision/models/SenseNova-Vision-7B-MoT}"
SENSENOVA_SOURCE_PATH="${SENSENOVA_SOURCE_PATH:-/data/nvme0/lhd_codes/SenseNova-Vision}"
CONFIG_JSON="${CONFIG_JSON:-${lightx2v_path}/configs/sensenova_vision/sensenova_vision.json}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
METRIC_PORT="${METRIC_PORT:-8001}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-10}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --metric-port)
            METRIC_PORT="$2"
            shift 2
            ;;
        --max-queue-size)
            MAX_QUEUE_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash $0 [--gpus 0] [--host 0.0.0.0] [--port 8000] [--metric-port 8001] [--max-queue-size 10]"
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! "${GPU_LIST}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: --gpus must be a comma-separated GPU index list, got '${GPU_LIST}'." >&2
    exit 2
fi
if [[ ! -d "${model_path}" ]]; then
    echo "Error: model_path does not exist: ${model_path}" >&2
    exit 2
fi
if [[ ! -f "${CONFIG_JSON}" ]]; then
    echo "Error: CONFIG_JSON does not exist: ${CONFIG_JSON}" >&2
    exit 2
fi

profiling_debug_level="${PROFILING_DEBUG_LEVEL:-0}"
enable_profiling_debug="${ENABLE_PROFILING_DEBUG:-false}"
recorder_mode="${RECORDER_MODE:-0}"

export lightx2v_path model_path
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
source "${lightx2v_path}/scripts/base/base.sh"

# base.sh is shared with benchmark scripts and enables verbose profiling by
# default. Restore the service-oriented values selected before sourcing it.
export PROFILING_DEBUG_LEVEL="${profiling_debug_level}"
export ENABLE_PROFILING_DEBUG="${enable_profiling_debug}"
export RECORDER_MODE="${recorder_mode}"
export LIGHTX2V_CACHE_DIR="${LIGHTX2V_CACHE_DIR:-${lightx2v_path}/save_results/sensenova_vision_server_cache}"
export LIGHTX2V_METRIC_PORT="${METRIC_PORT}"
mkdir -p "${LIGHTX2V_CACHE_DIR}"

echo "Starting one resident SenseNova-Vision model on physical GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "API: http://${HOST}:${PORT}/v1/tasks/sensenova-vision/"
echo "Artifacts: ${LIGHTX2V_CACHE_DIR}/outputs"

exec python -m lightx2v.server \
    --model_cls sensenova_vision \
    --task raw_query \
    --model_path "${model_path}" \
    --config_json "${CONFIG_JSON}" \
    --sensenova_source_path "${SENSENOVA_SOURCE_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --metric_port "${METRIC_PORT}" \
    --max_queue_size "${MAX_QUEUE_SIZE}"
