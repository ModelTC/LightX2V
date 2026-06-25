#!/bin/bash

# Flux2 Dev T2I 8-NPU tensor-parallel inference script for Ascend NPU 910B
# Usage: bash scripts/platforms/ascend_npu/dist_8/infer_flux2_dev.sh

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme1/wushuo/LightX2V}
model_path=${MODEL_PATH:-/data/nvme1/wushuo/hf_models/models/black-forest-labs/FLUX.2-dev}
npus=${NPUS:-8}

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --nproc_per_node="${npus}" -m lightx2v.infer \
    --model_cls flux2_dev \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/ascend_npu/dist_8/flux2_dev.json" \
    --prompt 'Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text "BFL Diffusers" on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom.' \
    --save_result_path "${lightx2v_path}/save_results/flux2_dev_dist8.png" \
    --seed 42
