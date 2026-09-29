#!/bin/bash
# MiniMax-H3 ref2av on 2/4/8 Radeon R9700 (gfx1201) GPUs with the aiter gfx1201 ops (radeon_gfx1201_h3).
#   HIP_VISIBLE_DEVICES=0,1             bash run_minimax_h3_ref2av_radeon_mgpu.sh --infer_steps 20  -> Ulysses SP2
#   HIP_VISIBLE_DEVICES=0,1,2,3         bash run_minimax_h3_ref2av_radeon_mgpu.sh --infer_steps 20  -> SP4
#   HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_minimax_h3_ref2av_radeon_mgpu.sh --infer_steps 20  -> SP8
# H3_LAYOUT=tp with 2 GPUs runs the TP2 fused path instead. Runtime features (radeon_runtime.py) are on by default;
# RADEON_CORESW_<NAME>=0 disables one. Inputs: PROMPT, IMAGE_PATH (comma separated), SAVE_RESULT_PATH, SEED.
set -euo pipefail
lightx2v_path="${LIGHTX2V_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
model_path="${MODEL_PATH:-/path/to/MiniMax-H3}"

STEPS=20
if [[ "${1:-}" == "--infer_steps" && -n "${2:-}" ]]; then
  STEPS="$2"
  shift 2
fi
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
IFS=, read -r -a DEVICES <<< "$HIP_VISIBLE_DEVICES"
NPROC="${#DEVICES[@]}"
LAYOUT="${H3_LAYOUT:-sp}"
if [[ "$LAYOUT" == tp && "$NPROC" != 2 ]] || [[ "$LAYOUT" == sp && ! "$NPROC" =~ ^(2|4|8)$ ]]; then
  echo "unsupported: layout=$LAYOUT with $NPROC GPUs (tp: 2, sp: 2/4/8)" >&2
  exit 2
fi
CFG="$(mktemp --suffix=.json)"
python3 - "$lightx2v_path/configs/minimax_h3/radeon_gfx1201/minimax_h3_ref2av_${LAYOUT}${NPROC}.json" "$CFG" "$STEPS" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["infer_steps"] = int(sys.argv[3])
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
PY

source "${lightx2v_path}/scripts/base/base.sh"
export PLATFORM=amd_rocm PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# FP32 VAE convolutions run on rocBLAS (RADEON_CORESW_FP32_CONV_ROCBLAS); keep rocBLAS off hipBLASLt for them.
export ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-0}"
# RCCL P2P defaults to 2 channels; 16 shortens the SP exchanges that still use RCCL.
export NCCL_MIN_P2P_NCHANNELS="${NCCL_MIN_P2P_NCHANNELS:-16}" NCCL_MAX_P2P_NCHANNELS="${NCCL_MAX_P2P_NCHANNELS:-16}"

CACHE_DIR="$(python3 -c "import json, os, sys; c = json.load(open(sys.argv[1])); print(os.path.expanduser(c['adaln_cache_dir']))" "$CFG")"
if ! ls -d "$CACHE_DIR"/minimax_h3/ref2av_$(printf "%02d" "$STEPS")steps_* >/dev/null 2>&1; then
  HIP_VISIBLE_DEVICES="${DEVICES[0]}" python3 "$lightx2v_path/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
    --model_path "$model_path" --config_json "$CFG" --model-variant ref2av
fi

torchrun --standalone --nproc_per_node="$NPROC" -m lightx2v.infer \
  --seed "${SEED:-42}" \
  --model_cls minimax_h3 \
  --model-variant ref2av \
  --task ref2av \
  --model_path "$model_path" \
  --config_json "$CFG" \
  --prompt "${PROMPT:-Generate an audio-video scene following the references.}" \
  --image_path "${IMAGE_PATH:-$lightx2v_path/assets/inputs/imgs/img_0.jpg}" \
  --size 768 1344 \
  --save_result_path "${SAVE_RESULT_PATH:-$lightx2v_path/save_results/minimax_h3_ref2av_${LAYOUT}${NPROC}.mp4}"
