#!/usr/bin/env bash

set -euo pipefail

H3_CONFIG_PATH="${H3_CONFIG_PATH:-configs/train/dmd/minimax_h3_t2av_dmd_lora.yaml}"
# The supplied config shards student, fake, and teacher across all eight GPUs.
# Override this when launching on a different node shape.
H3_NUM_PROCESSES="${H3_NUM_PROCESSES:-8}"

torchrun \
    --standalone \
    --nproc_per_node="${H3_NUM_PROCESSES}" \
    train.py \
    --config "${H3_CONFIG_PATH}"
