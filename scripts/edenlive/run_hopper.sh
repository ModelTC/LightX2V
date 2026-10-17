#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-/path/to/EdenLive-x2v}"
image_path="${IMAGE_PATH:-/path/to/cover.png}"
video_path="${VIDEO_PATH:-/path/to/recording.mp4}"
prompt="${PROMPT:-A cinematic scene with detailed, realistic lighting.}"
dit_ckpt="${DIT_QUANTIZED_CKPT:-$model_path/quantized/dit-5b-fp8.safetensors}"
t5_ckpt="${T5_QUANTIZED_CKPT:-$model_path/quantized/t5-fp8.safetensors}"
t5_offload="${T5_CPU_OFFLOAD:-false}"
tae_path="${TAE_PATH:-$model_path/decoder-5b/lighttaew2_2.pth}"
output_path="${SAVE_RESULT_PATH:-$repo_dir/save_results/edenlive_5b_hopper_fp8_sgl_24fps.mp4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 TOKENIZERS_PARALLELISM=false
export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# Resolve weights on the destination host without modifying the saved profile.
runtime_config="$(mktemp "${TMPDIR:-/tmp}/edenlive_hopper.XXXXXX.json")"
trap 'rm -f -- "$runtime_config"' EXIT
python - "$repo_dir/configs/edenlive/realtime/wan22_5b_ar_fp8_sgl_hopper.json" \
  "$runtime_config" "$dit_ckpt" "$tae_path" "$t5_ckpt" "$t5_offload" <<'PY'
import json
import sys
from pathlib import Path

source, output, dit, tae, t5, t5_offload = sys.argv[1:]
if t5_offload.lower() not in ("true", "false"):
    raise ValueError("T5_CPU_OFFLOAD must be true or false")
config = json.loads(Path(source).read_text())
config.update(
    dit_quantized_ckpt=str(Path(dit).resolve()),
    tae_path=str(Path(tae).resolve()),
    t5_quantized_ckpt=str(Path(t5).resolve()),
    t5_cpu_offload=t5_offload.lower() == "true",
)
Path(output).write_text(json.dumps(config, indent=4) + "\n")
PY

cd "$repo_dir"
python -m lightx2v.infer \
  --model_cls wan22_edenlive_ar --task i2v \
  --model_path "$model_path" --config_json "$runtime_config" \
  --prompt "$prompt" --image_path "$image_path" \
  --video_path "$video_path" --seed 2026092901 \
  --save_result_path "$output_path" "$@"
