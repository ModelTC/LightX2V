#!/usr/bin/env bash
# Fresh full-parameter three-way experiment, never attach to an older run.
set -Eeuo pipefail
train_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$train_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
case "${WAN_DMD_PRECISION_PROFILE:-fp32_sdpa}" in
  fp32_sdpa) export NVIDIA_TF32_OVERRIDE=0 ;;
  bf16_fa3) export NVIDIA_TF32_OVERRIDE=1 ;;
  *) printf 'Invalid WAN_DMD_PRECISION_PROFILE: %s\n' "$WAN_DMD_PRECISION_PROFILE" >&2; exit 2 ;;
esac
exec "${WAN_DMD_PYTHON:-python3}" -u "$train_root/scripts/launch_wan14b_comparison.py" "$@"
