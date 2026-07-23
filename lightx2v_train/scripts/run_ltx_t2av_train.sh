#!/bin/bash
# full parameters train use fsdp2 by default
export CUDA_VISIBLE_DEVICES=1,2,3,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
--standalone \
--nproc_per_node=4 \
train.py --config /data/nvme5/gushiqiao/codes/new/LightX2V/lightx2v_train/configs/train/dmd/ltx_t2av_ar_dmd_lora.yaml
