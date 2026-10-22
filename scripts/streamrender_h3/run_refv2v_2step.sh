#!/bin/bash
set -eo pipefail

lightx2v_path=/data/nvme6/gushiqiao/codes/report/LightX2V
model_path=/data/nvme5/gushiqiao/models/StreamRender-H3
example_root=/data/nvme5/gushiqiao/models/StreamRender-H3-examples
semantic_video=${SEMANTIC_VIDEO:-${example_root}/examples/media/videos/7622f0459a694082db1f.mp4}
reference_image=${REFERENCE_IMAGE:-${model_path}/reference_image/ref.png}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/streamrender_h3_refv2v_2step.mp4}
config_path=${CONFIG_PATH:-${lightx2v_path}/configs/streamrender_h3/refv2v_2step.json}
num_frames=${NUM_FRAMES:-107}
python_bin=${PYTHON_BIN:-python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

cd "${lightx2v_path}"
"${python_bin}" -m lightx2v.infer \
  --model_cls streamrender_h3 \
  --task refv2v \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --video_path "${semantic_video}" \
  --image_path "${reference_image}" \
  --num_frames "${num_frames}" \
  --seed 20260830 \
  --save_result_path "${output_path}"
