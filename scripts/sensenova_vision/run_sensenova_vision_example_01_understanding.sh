#!/usr/bin/env bash

# bash scripts/sensenova_vision/run_sensenova_vision_example_01_understanding.sh  --gpus 7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_sensenova_vision_example.sh" 01 "$@"
