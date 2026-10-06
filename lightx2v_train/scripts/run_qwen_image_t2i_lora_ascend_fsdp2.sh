#!/usr/bin/env bash
set -euo pipefail

CANN_HOME=/usr/local/Ascend/cann-8.5.1
export ASCEND_HOME_PATH="${CANN_HOME}" ASCEND_OPP_PATH="${CANN_HOME}/opp"
export PATH="${CANN_HOME}/bin:${CANN_HOME}/tools/ccec_compiler/bin:${PATH}"
export PYTHONPATH="${CANN_HOME}/python/site-packages:${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CANN_HOME}/lib64:${CANN_HOME}/lib64/plugin/opskernel:${CANN_HOME}/lib64/plugin/nnengine:${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --config configs/train/flow/qwen_image_lora_fsdp2_cache.yaml
