#!/bin/bash

export CUDA_VISIBLE_DEVICES=3,5,6,7

torchrun \
--standalone \
--nproc_per_node=4 \
train.py --config /data/nvme5/gushiqiao/codes/new/LightX2V/lightx2v_train/configs/train/dmd/lingbot_video_t2v_droid_dmd_lora.yaml
