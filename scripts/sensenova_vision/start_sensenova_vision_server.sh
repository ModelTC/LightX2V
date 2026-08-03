#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
model_path="${model_path:-/data/nvme0/lhd_codes/SenseNova-Vision/models/SenseNova-Vision-7B-MoT}"
SENSENOVA_SOURCE_PATH="${SENSENOVA_SOURCE_PATH:-/data/nvme0/lhd_codes/sensenova-vision-v2}"
CONFIG_JSON="${CONFIG_JSON:-${lightx2v_path}/configs/sensenova_vision/sensenova_vision.json}"
LIGHTX2V_CACHE_DIR="${LIGHTX2V_CACHE_DIR:-${lightx2v_path}/save_results/sensenova_vision_server_cache}"
OFFICIAL_PARITY="${SENSENOVA_OFFICIAL_PARITY:-true}"
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
        --model-path)
            model_path="$2"
            shift 2
            ;;
        --source-path)
            SENSENOVA_SOURCE_PATH="$2"
            shift 2
            ;;
        --config-json)
            CONFIG_JSON="$2"
            shift 2
            ;;
        --cache-dir)
            LIGHTX2V_CACHE_DIR="$2"
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
        --official-parity)
            OFFICIAL_PARITY="true"
            shift
            ;;
        --no-official-parity)
            OFFICIAL_PARITY="false"
            shift
            ;;
        -h|--help)
            echo "Usage: bash $0 [options]"
            echo "  --gpus INDEX[,INDEX]  --host HOST  --port PORT  --metric-port PORT"
            echo "  --model-path PATH  --source-path PATH  --config-json PATH  --cache-dir PATH"
            echo "  --max-queue-size N  --official-parity (default)  --no-official-parity"
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
if [[ "${OFFICIAL_PARITY}" != "true" && "${OFFICIAL_PARITY}" != "false" ]]; then
    echo "Error: SENSENOVA_OFFICIAL_PARITY must be true or false, got '${OFFICIAL_PARITY}'." >&2
    exit 2
fi
if [[ ! -d "${model_path}" ]]; then
    echo "Error: model_path does not exist: ${model_path}" >&2
    exit 2
fi
if [[ ! -d "${SENSENOVA_SOURCE_PATH}" ]]; then
    echo "Error: SenseNova-Vision source path does not exist: ${SENSENOVA_SOURCE_PATH}" >&2
    exit 2
fi
if [[ ! -f "${SENSENOVA_SOURCE_PATH}/inference/example_visualize.py" ]]; then
    echo "Error: official example_visualize.py is missing under: ${SENSENOVA_SOURCE_PATH}" >&2
    exit 2
fi
if [[ ! -f "${CONFIG_JSON}" ]]; then
    echo "Error: CONFIG_JSON does not exist: ${CONFIG_JSON}" >&2
    exit 2
fi

if [[ "${OFFICIAL_PARITY}" == "true" ]]; then
    profiling_debug_level="0"
    enable_profiling_debug="false"
    recorder_mode="0"
    export PYTHONHASHSEED="0"
else
    profiling_debug_level="${PROFILING_DEBUG_LEVEL:-0}"
    enable_profiling_debug="${ENABLE_PROFILING_DEBUG:-false}"
    recorder_mode="${RECORDER_MODE:-0}"
fi

export lightx2v_path model_path SENSENOVA_SOURCE_PATH
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"

# base.sh is shared with benchmark scripts and enables verbose profiling by
# default. Restore the service-oriented values selected before sourcing it.
export PROFILING_DEBUG_LEVEL="${profiling_debug_level}"
export ENABLE_PROFILING_DEBUG="${enable_profiling_debug}"
export RECORDER_MODE="${recorder_mode}"
export LIGHTX2V_CACHE_DIR
export LIGHTX2V_METRIC_PORT="${METRIC_PORT}"
if [[ "${OFFICIAL_PARITY}" == "true" ]]; then
    export DTYPE="BF16"
    export SENSITIVE_LAYER_DTYPE="None"
fi
mkdir -p "${LIGHTX2V_CACHE_DIR}"

echo "Starting one resident SenseNova-Vision model on physical GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "Official parity mode: ${OFFICIAL_PARITY}"
echo "Model: ${model_path}"
echo "Official source: ${SENSENOVA_SOURCE_PATH}"
echo "Config: ${CONFIG_JSON}"
echo "API: http://${HOST}:${PORT}/v1/tasks/sensenova-vision/"
echo "Artifacts: ${LIGHTX2V_CACHE_DIR}/outputs"
if [[ "${GPU_LIST}" == *,* ]]; then
    echo "Warning: SenseNova-Vision is not sharded; the resident model uses the first visible GPU." >&2
fi

exec python -m lightx2v.server \
    --model_cls sensenova_vision \
    --task omni_vision_task \
    --model_path "${model_path}" \
    --config_json "${CONFIG_JSON}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --metric_port "${METRIC_PORT}" \
    --max_queue_size "${MAX_QUEUE_SIZE}"
