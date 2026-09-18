#!/bin/bash
set -e

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Qwen-Image-2512

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

python -m torch.distributed.run --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls qwen_image \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/qwen_image/offload/qwen_image_2512_block_shared.json" \
  --shared_cpu_weight_scope host \
  --prompt 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.' \
  --negative_prompt " " \
  --save_result_path "${lightx2v_path}/save_results/qwen_image_t2i_2512_block_shared_offload.png" \
  --seed 42
