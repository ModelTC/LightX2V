#!/bin/bash
set -eo pipefail

lightx2v_path=/path/to/LightX2V
model_path=/path/to/LongCat-Video-Avatar-1.6

export CUDA_VISIBLE_DEVICES=0,1,2,3
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16

python -m torch.distributed.run --standalone --nproc_per_node=4 -m lightx2v.server \
  --model_cls longcat_video_avatar \
  --task s2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/longcat_video_avatar/h800/s2v_bf16_sp4.json" \
  --host 0.0.0.0 \
  --port 8000
