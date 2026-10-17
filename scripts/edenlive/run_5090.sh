#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
vast_dir="$(dirname -- "$repo_dir")"
gpu_count="${GPU_COUNT:-1}"
case "$gpu_count" in
  1) default_devices=0; profile=wan22_5b_ar_fp8_lighttae_cached.json; launch=(python) ;;
  2) default_devices=0,1; profile=wan22_5b_ar_fp8_lighttae_sp2.json; launch=(torchrun --standalone --nproc_per_node=2) ;;
  4) default_devices=0,1,2,3; profile=wan22_5b_ar_fp8_lighttae_sp4.json; launch=(torchrun --standalone --nproc_per_node=4) ;;
  *) echo "GPU_COUNT must be 1, 2, or 4" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_devices}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 TOKENIZERS_PARALLELISM=false
export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
sample_dir="$vast_dir/EdenLive-Official-wp-demo/examples/samples/cryo-crossing"
prompt="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["staticPrompt"])' "$sample_dir/prompts.json")"
cd "$repo_dir"
"${launch[@]}" -m lightx2v.infer \
  --model_cls wan22_edenlive_ar --task i2v \
  --model_path "$vast_dir/EdenLive-x2v" \
  --config_json "$repo_dir/configs/edenlive/5090/$profile" \
  --prompt "$prompt" --image_path "$sample_dir/cover.png" \
  --video_path "$sample_dir/recording.mp4" --seed 2026092901 \
  --save_result_path "$repo_dir/save_results/edenlive_5090_${gpu_count}gpu_16fps.mp4" "$@"
