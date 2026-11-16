#!/bin/bash

# AdaLN cache setup:
# The default inference JSON configs enable "use_adaln_cache": true.
# Generate the cache before server startup; skip this step if a matching cache exists.
# 1. Set lightx2v_path, model_path, --config_json, and platform/device settings in
#    tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh.
# 2. From the repository root, run:
#    bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_image_adaln.sh
# 3. Start the server after cache generation completes.
# Cache generation and inference must use the same model weights, timesteps,
# platform, and adaln_cache_dir. Point --config_json to the inference JSON.
# One GPU generates both T2I and I2I cache profiles, reusable across TP/SP configs.
# The tool will not overwrite an existing cache directory; move it aside to rebuild.

lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3-Image

export CUDA_VISIBLE_DEVICES=0,1,2,3
source "${lightx2v_path}/scripts/base/base.sh"

export LIGHTX2V_CACHE_DIR="${lightx2v_path}/save_results/server_cache"

# DiT TP2 x SP2, Qwen3-VL TP4, and VAE tile parallelism share four GPUs.
# Leave --task unset to serve both t2i and i2i requests with the same POST clients.
torchrun --standalone --nproc_per_node=4 -m lightx2v.server \
  --model_cls minimax_h3_image \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_image/minimax_h3_image_tp_sp.json" \
  --host 0.0.0.0 \
  --port 8000
