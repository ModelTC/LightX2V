#!/bin/bash
set -eo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MINIMAX_H3_MODEL_PATH:-${lightx2v_path}/models/MiniMax-H3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
source "${lightx2v_path}/scripts/base/base.sh"

# Ref2AV requires transformer_ref/ and its separate 29-step Ref2AV AdaLN cache.
# Reference lists use comma-separated paths. Audio needs an image or video reference.
image_path="${MINIMAX_H3_IMAGE_PATH:-}"
video_path="${MINIMAX_H3_VIDEO_PATH:-}"
audio_path="${MINIMAX_H3_AUDIO_PATH:-}"
if [[ -z "${image_path}" && -z "${video_path}" && -z "${audio_path}" ]]; then
  image_path="${lightx2v_path}/assets/inputs/imgs/img_0.jpg"
fi
if [[ -z "${image_path}" && -z "${video_path}" ]]; then
  echo "Ref2AV requires an image or video reference; audio-only input is unsupported." >&2
  exit 1
fi
reference_args=()
if [[ -n "${image_path}" ]]; then
  reference_args+=(--image_path "${image_path}")
fi
if [[ -n "${video_path}" ]]; then
  reference_args+=(--video_path "${video_path}")
fi
if [[ -n "${audio_path}" ]]; then
  reference_args+=(--audio_path "${audio_path}")
fi

exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task ref2av \
  --model_path "${model_path}" \
  --config_json "${MINIMAX_H3_CONFIG:-${lightx2v_path}/configs/minimax_h3/offload/minimax_h3_t2av_block_shared_offload_numa_sp8.json}" \
  --prompt "${MINIMAX_H3_PROMPT:-Generate an audio-video scene following the references.}" \
  "${reference_args[@]}" \
  --save_result_path "${MINIMAX_H3_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/minimax_h3_ref2av_block_shared_offload_numa_sp8.mp4}" \
  --seed 42
