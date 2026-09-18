#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/data/wushuo/sparse_update/LightX2V-sparse}
model_path=${MODEL_PATH:-/data/MiniMax-H3}
case_name=${CASE_NAME:-case1}

case "${case_name}" in
  case1)
    case_path=/data/wushuo/商汤H3测试数据/Case1_mecha1
    prompt_path=${case_path}/prompt_h3.txt
    image_path=${case_path}/Picture_1.jpg,${case_path}/Picture_2.png,${case_path}/Picture_3.jpg,${case_path}/Picture_4.png
    height=768
    width=1344
    num_frames=294
    ;;
  penpen)
    case_path=/data/wushuo/商汤H3测试数据/penpen_CASE
    prompt_path=${case_path}/prompt.txt
    image_path=${case_path}/01_character.png,${case_path}/02_ui_style.png
    height=736
    width=1280
    num_frames=362
    ;;
  *)
    echo "CASE_NAME must be case1 or penpen, got: ${case_name}" >&2
    exit 2
    ;;
esac

prompt=$(<"${prompt_path}")
prompt=${prompt//$'\r'/}
if [ "${case_name}" = penpen ]; then
  prompt=${prompt//$'\\n'/$'\n'}
  prompt=${prompt//$'\\"'/'"'}
fi

config_path=${lightx2v_path}/configs/platforms/mthreads_musa/minimax_h3_ref2av_fl2v_lora8_tp8_moorcat_h3_sparse.json
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/${case_name}_minimax_h3_ref2av_fl2v_lora8_tp8_moorcat_h3_sparse.mp4}

export PLATFORM=musa
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=, read -r -a musa_devices <<< "${MUSA_VISIBLE_DEVICES}"
if [ "${#musa_devices[@]}" -ne 8 ]; then
  echo "MUSA_VISIBLE_DEVICES must contain 8 devices, got: ${MUSA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES}
export PYTHONPATH=${MOORCAT_PYTHONPATH:-/data/wushuo/LightX2V-0905/third_party}:${PYTHONPATH:-}
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

source "${lightx2v_path}/scripts/base/base.sh"
python -c "from moorcat.blocksparse import block_sparse_attn_func_indexed_fast; from flash_attn_interface import flash_attn_varlen_func"

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task ref2av \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --prompt "${prompt}" \
  --image_path "${image_path}" \
  --size "${height}" "${width}" \
  --num_frames "${num_frames}" \
  --save_result_path "${output_path}" \
  --seed 42
