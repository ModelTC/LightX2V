#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 MODEL_PATH INPUT_IMAGE_OR_VIDEO OUTPUT_PNG_OR_MP4" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="$(cd -- "${script_dir}/../.." && pwd)"
model_path="$1"
input_path="$2"
output_path="$3"

case "${output_path,,}" in
    *.png) input_flag="--image_path" ;;
    *.mp4) input_flag="--video_path" ;;
    *) echo "Output must end in .png or .mp4" >&2; exit 2 ;;
esac

export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -B -m lightx2v.infer \
    --model_cls depth_anything3 \
    --model-variant mono_large \
    --task depth \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/depth_anything3/da3mono_large.json" \
    "${input_flag}" "${input_path}" \
    --save_result_path "${output_path}"
