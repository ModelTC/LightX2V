#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 MODEL_PATH IMAGE_PATH OUTPUT_PNG" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="$(cd -- "${script_dir}/../.." && pwd)"
model_path="$1"
image_path="$2"
output_path="$3"

export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -B -m lightx2v.infer \
    --model_cls depth_anything3 \
    --model-variant mono_large \
    --task depth \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/depth_anything3/da3mono_large.json" \
    --image_path "${image_path}" \
    --save_result_path "${output_path}"
