#!/bin/bash
set -euo pipefail

lightx2v_path=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_path=$(realpath "${MODEL_PATH:-${lightx2v_path}/../models/H3/StreamRender-H3}")
example_root=${EXAMPLE_ROOT:-${lightx2v_path}/../models/H3/StreamRender-H3-examples}
semantic_video=${SEMANTIC_VIDEO:-${example_root}/examples/media/videos/7622f0459a694082db1f.mp4}
reference_image=${REFERENCE_IMAGE:-${model_path}/reference_image/ref.png}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/streamrender_h3_refv2v_2step.mp4}
config_path=${CONFIG_PATH:-${lightx2v_path}/configs/streamrender_h3/refv2v_107f_bf16_sage_5090_tp4_sp2.json}
num_frames=${NUM_FRAMES:-107}
python_bin=${PYTHON_BIN:-python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PROFILING_DEBUG_LEVEL=${PROFILING_DEBUG_LEVEL:-0}
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

cd "${lightx2v_path}"
IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node="${#visible_devices[@]}" \
  -m lightx2v.infer \
  --model_cls streamrender_h3 \
  --task refv2v \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --video_path "${semantic_video}" \
  --image_path "${reference_image}" \
  --num_frames "${num_frames}" \
  --seed 20260830 \
  --save_result_path "${output_path}"
