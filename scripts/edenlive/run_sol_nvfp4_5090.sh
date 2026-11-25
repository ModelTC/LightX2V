#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
vast="$(dirname "$repo")"
model="${EDEN_MODEL_PATH:-$vast/EdenLive-x2v}"
data="${EDEN_REPRO_DATA:-$vast/edenlive_repro}"
output="${EDEN_OUTPUT_DIR:-$data/run}"
lightvae="${EDEN_LIGHTVAE_REPO:-$vast/LightVAE}"
export PYTHONPATH="$repo:$data/deps:$lightvae:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16 PROFILING_DEBUG_LEVEL=0 OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$repo"
python scripts/edenlive/benchmark_sol_nvfp4_5090.py "$model" "$data" "$output"
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of csv=p=0 "$output/wan22_5b_ar_sol_nvfp4_lightvae_5090.mp4"
