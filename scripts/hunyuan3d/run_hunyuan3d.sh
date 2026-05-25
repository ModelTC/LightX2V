#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Hunyuan3D-2.1

# Hunyuan3D-2.1 full pipeline: image -> shape mesh (.glb) -> textured mesh (.glb)
# 1. git clone Hunyuan3D-2.1 and install following readme，use “pip install --no-cache-dir --no-build-isolation -v -e . ” to compile（pybind11==2.13.4 is need）
# 2. ln -sfn /path/to/Hunyuan3D-2.1/hy3dpaint ${lightx2v_path}/tools/postprocess/hy3dpaint

export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=FP16
export hy_repo=/path/to/Hunyuan3D-2.1

image_path=${hy_repo}/assets/demo.png
output_dir=${lightx2v_path}/save_results/hunyuan3d
mesh_path=${output_dir}/demo.glb
textured_path=${output_dir}/demo_textured.glb

mkdir -p "${output_dir}"

echo "=== Step 1/2: shape generation ==="
python -m lightx2v.infer \
    --model_cls hunyuan3d \
    --task i23d \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/hunyuan3d/hunyuan3d_shape.json" \
    --image_path "${image_path}" \
    --save_result_path "${mesh_path}" \
    --seed 42

echo "Saved mesh: ${mesh_path}"

echo "=== Step 2/2: mesh texture (paint) ==="
python ${lightx2v_path}/tools/postprocess/postprocess_paint.py \
    --model_path "${model_path}" \
    --mesh_path "${mesh_path}" \
    --image_path "${image_path}" \
    --save_path "${textured_path}" \
    --max_num_view 6 \
    --resolution 512

echo "Saved textured mesh: ${textured_path}"
echo "All outputs in: ${output_dir}"
