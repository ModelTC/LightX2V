#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
vast_dir="$(dirname -- "$repo_dir")"
model_path="${MODEL_PATH:-$vast_dir/EdenLive-x2v}"
sample_dir="${SAMPLE_DIR:-$vast_dir/EdenLive-Official-wp-demo/examples/samples/cryo-crossing}"
config_json="${CONFIG_JSON:-$repo_dir/configs/edenlive/wan22_5b_few_step.json}"
output_path="${SAVE_RESULT_PATH:-$repo_dir/save_results/edenlive_5b_few_step.mp4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 TOKENIZERS_PARALLELISM=false
export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

prompt="${PROMPT:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["staticPrompt"])' "$sample_dir/prompts.json")}"

cd "$repo_dir"
python -m lightx2v.infer \
  --model_cls wan22_edenlive_ar --task i2v \
  --model_path "$model_path" --config_json "$config_json" \
  --prompt "$prompt" \
  --image_path "$sample_dir/cover.png" --video_path "$sample_dir/recording.mp4" \
  --seed 2026092901 --save_result_path "$output_path" "$@"
