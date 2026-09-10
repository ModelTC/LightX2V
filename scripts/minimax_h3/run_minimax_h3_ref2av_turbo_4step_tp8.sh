#!/bin/bash

set -euo pipefail

lightx2v_path="${LIGHTX2V_PATH:-/data/wushuo1/LightX2V}"
model_path="${MODEL_PATH:-/data/wushuo1/models/wyr_models/minimax_h3/h3_hf_bf16}"
test_case_path="${TEST_CASE_PATH:-/data/wushuo1/商汤H3测试数据/Case1_mecha1}"
config_path="${CONFIG_PATH:-${lightx2v_path}/configs/minimax_h3/dmd/minimax_h3_ref2av_turbo_4step_tp8.json}"
output_path="${OUTPUT_PATH:-${test_case_path}/output_minimax_h3_ref2av_turbo_4step_bf16.mp4}"
log_path="${LOG_PATH:-${test_case_path}/run_minimax_h3_ref2av_turbo_4step_bf16_tp8.log}"

export PLATFORM="${PLATFORM:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PYTHONPATH="${PYTHONPATH:-}"

source "${lightx2v_path}/scripts/base/base.sh"

cache_manifest="/data/wushuo1/.cache/lightx2v/adaln/minimax_h3/ref2av_04steps/manifest.json"
if [[ ! -f "${cache_manifest}" ]]; then
  CUDA_VISIBLE_DEVICES=0 python "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
    --model_path "${model_path}" \
    --config_json "${config_path}" \
    --task ref2av
fi

prompt="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text(encoding="utf-8").strip(), end="")' "${test_case_path}/prompt_h3.txt")"
image_path="${test_case_path}/Picture_1.jpg,${test_case_path}/Picture_2.png,${test_case_path}/Picture_3.jpg,${test_case_path}/Picture_4.png"

mkdir -p "$(dirname "${output_path}")" "$(dirname "${log_path}")"

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task ref2av \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --prompt "${prompt}" \
  --image_path "${image_path}" \
  --save_result_path "${output_path}" \
  --seed 42 \
  2>&1 | tee "${log_path}"
