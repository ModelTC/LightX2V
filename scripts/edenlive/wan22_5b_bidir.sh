#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-/path/to/EdenLive-x2v}"
image_path="${IMAGE_PATH:-/path/to/cover.png}"
video_path="${VIDEO_PATH:-/path/to/recording.mp4}"
prompt="${PROMPT:-A cinematic scene with detailed, realistic lighting.}"
config_json="${CONFIG_JSON:-$repo_dir/configs/edenlive/wan22_5b_bidir.json}"
output_path="${SAVE_RESULT_PATH:-$repo_dir/save_results/edenlive_5b_bidir.mp4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 TOKENIZERS_PARALLELISM=false
export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多'
cd "$repo_dir"
python -m lightx2v.infer \
  --model_cls wan22_edenlive --task i2v \
  --model_path "$model_path" --config_json "$config_json" \
  --prompt "$prompt" --negative_prompt "${NEGATIVE_PROMPT:-$negative_prompt}" \
  --image_path "$image_path" --video_path "$video_path" \
  --seed 42 --save_result_path "$output_path" "$@"
