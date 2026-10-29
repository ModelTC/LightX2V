#!/usr/bin/env bash
set -Eeuo pipefail
code_root="${WAN_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export WAN_TURBO_ROOT="${WAN_TURBO_ROOT:-/data/nvme6/gushiqiao/outputs/wan21_1_3b_turbo8_fp32_fsdp01_smoke_20261028}"
export WAN_TURBO_DATA_ROOT="${WAN_TURBO_DATA_ROOT:-/data/nvme6/gushiqiao/outputs/wan21_1_3b_turbo8_fp32_gpu0_smoke_20261028}"
export WAN_TURBO_MODEL="${WAN_TURBO_MODEL:-/data/nvme0/gushiqiao/models/official_models/Wan2.1-T2V-1.3B}"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$code_root/lightx2v_train:$code_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
config="${WAN_TURBO_CONFIG:-$code_root/lightx2v_train/configs/train/turbo/wan2_1_t2v_1_3b_turbo8_fp32_fsdp01_smoke.yaml}"
cd "$code_root/lightx2v_train"
exec "${WAN_PYTHON:-python}" -u -m torch.distributed.run --standalone --nproc_per_node=2 scripts/run_wan_turbo_fp32.py --config "$config"
