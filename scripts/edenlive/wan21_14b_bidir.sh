#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/EdenLive-x2v

export CUDA_VISIBLE_DEVICES=0
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
  --model_cls wan21_edenlive \
  --task i2v \
  --model_path "$model_path" \
  --config_json "${lightx2v_path}/configs/edenlive/wan21_14b_bidir.json" \
  --prompt 'Cinematic photorealism. A vast volcanic cavern dominated by a churning, incandescent molten lava lake, dark basalt formations, and jagged stalactites. Dynamic heat haze and glowing orange embers fill the air. Seen from a first-person viewpoint, a heavy industrial cryo-blaster with frosted metallic plating and cyan energy vents is held ready. Projectiles flash-freeze bubbling magma into dense slabs of hissing glacial ice. Dramatic lighting balances intense fiery magma radiance against cool cyan cryo luminescence.' \
  --negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  --image_path /data/nvme6/gushiqiao/codes/vast/EdenLive-Official-wp-demo/examples/samples/cryo-crossing/cover.png \
  --video_path /data/nvme6/gushiqiao/codes/vast/EdenLive-Official-wp-demo/examples/samples/cryo-crossing/recording.mp4 \
  --seed 42 \
  --save_result_path "${lightx2v_path}/save_results/edenlive_14b_bidir.mp4"
