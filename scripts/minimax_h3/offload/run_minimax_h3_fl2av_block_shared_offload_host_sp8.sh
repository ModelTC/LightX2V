#!/bin/bash
set -eo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MINIMAX_H3_MODEL_PATH:-${lightx2v_path}/models/MiniMax-H3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
source "${lightx2v_path}/scripts/base/base.sh"

# Reuse the 29-step FL2AV AdaLN cache shared by the four base tasks.
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task fl2av \
  --model_path "${model_path}" \
  --config_json "${MINIMAX_H3_CONFIG:-${lightx2v_path}/configs/minimax_h3/offload/minimax_h3_t2av_block_shared_offload_host_sp8.json}" \
  --prompt "${MINIMAX_H3_PROMPT:-Create a coherent transition with natural synchronized sound.}" \
  --image_path "${MINIMAX_H3_IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png}" \
  --last_frame_path "${MINIMAX_H3_LAST_FRAME_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png}" \
  --save_result_path "${MINIMAX_H3_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/minimax_h3_fl2av_block_shared_offload_host_sp8.mp4}" \
  --seed 42
