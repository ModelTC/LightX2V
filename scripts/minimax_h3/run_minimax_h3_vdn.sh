#!/usr/bin/env bash
set -eo pipefail
lightx2v_path=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_path=${H3_MODEL_PATH:-/data/nvme1/models/MiniMaxAI/MiniMax-H3}
export CUDA_VISIBLE_DEVICES=4,5,6,7
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
source "${lightx2v_path}/scripts/base/base.sh"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd "${lightx2v_path}"
python -m torch.distributed.run --standalone --nproc_per_node=4 -m lightx2v.infer \
  --model_cls minimax_h3 --model-variant fl2av --task t2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/vdn/vdn_sp_8step.json" \
  --prompt "A quiet forest stream at sunrise, with birdsong and flowing water." \
  --seed 42 \
  --save_result_path "${lightx2v_path}/save_results/output_minimax_h3_vdn.mp4" \
  "$@"
