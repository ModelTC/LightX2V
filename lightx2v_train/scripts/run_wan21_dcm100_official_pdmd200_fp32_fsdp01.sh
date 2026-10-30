#!/usr/bin/env bash
# Sequential dCM -> official PDMD, with independent checkpoints and optimizers.
set -Eeuo pipefail
code_root="${WAN_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export WAN_DCM_PDMD_ROOT="${WAN_DCM_PDMD_ROOT:-/data/nvme6/gushiqiao/outputs/wan21_1_3b_dcm100_official_pdmd200_fp32_fsdp01_20261030}"
export WAN_DCM_PDMD_DATA_ROOT="${WAN_DCM_PDMD_DATA_ROOT:-/data/nvme6/gushiqiao/outputs/wan21_1_3b_turbo8_fp32_gpu0_smoke_20261028}"
export WAN_DCM_PDMD_MODEL="${WAN_DCM_PDMD_MODEL:-/data/nvme0/gushiqiao/models/official_models/Wan2.1-T2V-1.3B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$code_root/lightx2v_train:$code_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$code_root/lightx2v_train"
exec "${WAN_PYTHON:-python}" -u scripts/run_wan21_dcm_pdmd_stages.py \
  --dcm-config "$code_root/lightx2v_train/configs/train/turbo/wan2_1_t2v_1_3b_dcm100_fp32_fsdp01.yaml" \
  --pdmd-config "$code_root/lightx2v_train/configs/train/dmd/wan2_1_t2v_1_3b_official_pdmd200_from_dcm_fp32_fsdp01.yaml" "$@"
