#!/bin/bash

# Z-Image Turbo T2I 8-GPU Inference Script for NVIDIA
# System management interface: nvidia-smi

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme1/wushuo/LightX2V}
model_path=${MODEL_PATH:-/data/nvme1/wushuo/hf_models/models/Z-Image-Turbo}
gpus=${GPUS:-8}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/nvidia/dist_8/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --nproc_per_node="${gpus}" -m lightx2v.infer \
    --model_cls z_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/nvidia/dist_8/z_image_turbo_t2i.json" \
    --prompt 'Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.' \
    --negative_prompt " " \
    --save_result_path "${lightx2v_path}/save_results/z_image_turbo_dist8.png" \
    --seed 42 \
    --aspect_ratio "16:9"
