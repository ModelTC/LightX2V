#!/bin/bash
# WorldMirror 3D reconstruction (FP8, single GPU)
set -e

lightx2v_path=/path/to/LightX2V
model_path=/path/to/HY-World-2.0

export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls worldmirror \
    --task recon \
    --model_path ${model_path} \
    --config_json ${lightx2v_path}/configs/worldmirror/worldmirror_recon_fp8.json \
    --input_path /path/to/scene_dir \
    --save_result_path ${lightx2v_path}/save_results/worldmirror \
    --save_rendered \
    --render_interp_per_pair 15
