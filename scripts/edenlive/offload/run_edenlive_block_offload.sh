#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
vast_dir="$(dirname -- "$repo_dir")"
model_path="${MODEL_PATH:-$vast_dir/EdenLive-x2v}"
sample_dir="${SAMPLE_DIR:-$vast_dir/EdenLive-Official-wp-demo/examples/samples/cryo-crossing}"
output_path="${SAVE_RESULT_PATH:-$repo_dir/save_results/edenlive_block_offload.mp4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# Derive offload settings from the retained Wan2.1 14B AR base profile.
runtime_config="$(mktemp "${TMPDIR:-/tmp}/edenlive_block_offload.XXXXXX.json")"
trap 'rm -f -- "$runtime_config"' EXIT
python - "$repo_dir/configs/edenlive/wan21_14b_few_step.json" "$runtime_config" <<'PY'
import json
import sys
from pathlib import Path

source, output = sys.argv[1:]
config = json.loads(Path(source).read_text())
config.update(
    num_frames=121,
    cpu_offload=True,
    offload_granularity="block",
    lazy_load=False,
    use_compile=True,
    warmup=True,
)
Path(output).write_text(json.dumps(config, indent=4) + "\n")
PY
prompt="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["staticPrompt"])' "$sample_dir/prompts.json")"

cd "$repo_dir"
python -m lightx2v.infer \
  --model_cls wan21_edenlive_ar --task i2v \
  --model_path "$model_path" --config_json "$runtime_config" \
  --prompt "$prompt" --image_path "$sample_dir/cover.png" \
  --video_path "$sample_dir/recording.mp4" --seed 2026092901 \
  --save_result_path "$output_path" "$@"
