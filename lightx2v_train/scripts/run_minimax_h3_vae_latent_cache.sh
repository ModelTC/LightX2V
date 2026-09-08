#!/usr/bin/env bash

set -euo pipefail

# Encode the original H3 teacher; the historical cache paths remain unchanged.

H3_TURBO_VAE_CACHE_GPUS="${H3_TURBO_VAE_CACHE_GPUS:-8}"
H3_TURBO_VAE_SOURCE="${H3_TURBO_VAE_SOURCE:-/data/nvme6/gushiqiao/datasets/minimax_h3_teacher_api_5s/metadata.jsonl}"
H3_TURBO_VAE_CACHE="${H3_TURBO_VAE_CACHE:-/data/nvme6/gushiqiao/datasets/minimax_h3_turbo_vae_latent_cache}"
H3_MODEL_PATH="${H3_MODEL_PATH:-/data/nvme6/gushiqiao/models/MiniMax-H3}"
H3_TURBO_VAE_CACHE_DTYPE="${H3_TURBO_VAE_CACHE_DTYPE:-fp32}"

torchrun \
    --standalone \
    --nproc_per_node="${H3_TURBO_VAE_CACHE_GPUS}" \
    tools/cache_minimax_h3_vae_latents.py \
    --input "${H3_TURBO_VAE_SOURCE}" \
    --output-dir "${H3_TURBO_VAE_CACHE}" \
    --model-path "${H3_MODEL_PATH}" \
    --posterior mode \
    --storage-dtype "${H3_TURBO_VAE_CACHE_DTYPE}"
