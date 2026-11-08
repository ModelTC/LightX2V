#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Requires a BF16-capable NVIDIA GPU and the flash-attn (v2) package.
export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CONFIG_JSON="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/nvidia/vggt_recon_bf16.json}"

exec bash "${lightx2v_path}/scripts/vggt/run_vggt.sh" "$@"
