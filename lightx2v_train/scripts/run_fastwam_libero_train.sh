#!/usr/bin/env bash
set -euo pipefail

LIGHTX2V_ROOT="${LIGHTX2V_ROOT:-/data/nvme0/chendingyu/LightX2V}"
CONFIG_PATH="${CONFIG_PATH:-${LIGHTX2V_ROOT}/lightx2v_train/configs/train/fastwam/libero_uncond_2cam224.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="${LIGHTX2V_ROOT}/lightx2v_train:${LIGHTX2V_ROOT}:${PYTHONPATH:-}"

cd "${LIGHTX2V_ROOT}"
python lightx2v_train/train.py --config "${CONFIG_PATH}" "$@"
