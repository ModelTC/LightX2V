#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

source_dir=${SOURCE_DIR:-/llm/models/MiniMax-H3/transformer}
output_dir=${OUTPUT_DIR:-/llm/models/MiniMax-H3/quantized/int8}

python "${REPO_ROOT}/tools/convert/converter.py" \
  --source "${source_dir}" \
  --output "${output_dir}" \
  --output_name minimax_h3_int8 \
  --model_type h3 \
  --quantized \
  --linear_type int8 \
  --device cpu \
  --single_file
