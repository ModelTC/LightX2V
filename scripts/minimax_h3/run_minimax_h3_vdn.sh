#!/bin/bash

# AdaLN cache setup:
# If the inference JSON config enables "use_adaln_cache": true, generate the cache before inference:
# 1. Set lightx2v_path, model_path, --config_json, and --model-variant in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh.
# 2. Use --model-variant fl2av and the VDN inference JSON config.
# 3. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
# Cache generation and inference must use the same JSON config and adaln_cache_dir.

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# TP8: vdn_tp_8step.json; SP8: vdn_sp_8step.json; TP4 + SP2: vdn_tp_sp_8step.json.
python -m torch.distributed.run --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/vdn/vdn_sp_8step.json" \
  --prompt 'A quiet forest stream at sunrise, with birdsong and flowing water.' \
  --save_result_path "${lightx2v_path}/save_results/output_minimax_h3_vdn.mp4" \
  --seed 42
