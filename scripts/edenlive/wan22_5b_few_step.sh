#!/bin/bash

lightx2v_path=/mnt/devsft_afs_2/gushiqiao/vast/LightX2V
model_path=/mnt/devsft_afs_2/gushiqiao/vast/EdenLive-x2v


export CUDA_VISIBLE_DEVICES=0
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
source ${lightx2v_path}/scripts/base/base.sh

export PROFILING_DEBUG_LEVEL=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

python -m lightx2v.infer \
  --model_cls wan22_edenlive_ar \
  --task i2v \
  --model_path "$model_path" \
  --config_json /mnt/devsft_afs_2/gushiqiao/vast/LightX2V/configs/edenlive/5090/wan22_5b_ar_int8_convrot_proxy_int8_lighttae.json \
  --prompt 'Cinematic photorealism. A vast volcanic cavern dominated by a churning, incandescent molten lava lake, dark basalt formations, and jagged stalactites. Dynamic heat haze and glowing orange embers fill the air. Seen from a first-person viewpoint, a heavy industrial cryo-blaster with frosted metallic plating and cyan energy vents is held ready. Projectiles flash-freeze bubbling magma into dense slabs of hissing glacial ice. Dramatic lighting balances intense fiery magma radiance against cool cyan cryo luminescence.' \
  --image_path /mnt/devsft_afs_2/gushiqiao/vast/EdenLive-Official-wp-demo/examples/samples/cryo-crossing/cover.png \
  --video_path /mnt/devsft_afs_2/gushiqiao/vast/EdenLive-Official-wp-demo/examples/samples/cryo-crossing/recording.mp4 \
  --seed 2026092901 \
  --save_result_path "${lightx2v_path}/save_results/edenlive_5b_5090_sla.mp4"
