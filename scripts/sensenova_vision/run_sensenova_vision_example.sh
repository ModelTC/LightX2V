#!/usr/bin/env bash
set -euo pipefail

# usage: bash scripts/sensenova_vision/run_sensenova_vision_example.sh all --gpus 7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 && "${1}" != --* ]]; then
    export EXAMPLE_ID="${1}"
    shift
else
    export EXAMPLE_ID="${EXAMPLE_ID:-all}"
fi

while [[ $# -gt 0 ]]; do
    case "${1}" in
        --gpus)
            if [[ $# -lt 2 ]]; then
                echo "Error: --gpus requires a GPU list such as 0 or 0,1." >&2
                exit 2
            fi
            export CUDA_VISIBLE_DEVICES="${2}"
            shift 2
            ;;
        --gpus=*)
            export CUDA_VISIBLE_DEVICES="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: bash ${BASH_SOURCE[0]} [all|01-14] [--gpus GPU_LIST]"
            echo "Example: bash ${BASH_SOURCE[0]} 03 --gpus 2"
            echo "Example: bash ${BASH_SOURCE[0]} 03 --gpus 2,3"
            exit 0
            ;;
        *)
            echo "Error: unknown argument: ${1}" >&2
            exit 2
            ;;
    esac
done

if [[ ! "${CUDA_VISIBLE_DEVICES:-0}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: --gpus must be a comma-separated GPU index list, got '${CUDA_VISIBLE_DEVICES}'." >&2
    exit 2
fi

exec bash "${SCRIPT_DIR}/run_sensenova_vision.sh" example
