#!/bin/bash
#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme4/gushiqiao/new/LightX2V
model_path=/data/nvme4/models/mgv2

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1_sf_mtxg2 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/matrix_game2/matrix_game2_universal.json \
--prompt '' \
--image_path /data/nvme4/gushiqiao/0007.png \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_matrix_game2_universal.mp4 \
--seed 42
