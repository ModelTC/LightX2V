#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/HunyuanImage-3-Instruct
export HUNYUAN_IMAGE3_REPO_PATH=/path/to/HunyuanImage-3.0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"

python -m torch.distributed.run --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls hunyuan_image3 \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/hunyuan_image3/offload/hunyuan_image3_block_shared.json" \
  --shared_cpu_weight_scope host \
  --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
  --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i_block_shared_offload.png" \
  --seed 42
