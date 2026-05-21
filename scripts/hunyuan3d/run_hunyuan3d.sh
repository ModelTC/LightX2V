#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Hunyuan3D-2.1

# Hunyuan3D-2.1 full pipeline: image -> shape mesh (.glb) -> textured mesh (.glb)
#
# ---- Paint prerequisites (one-time setup) ----
# Paint reuses upstream hy3dpaint from Hunyuan3D-2.1 (not copied into LightX2V).
# HF model weights stay in model_path; do NOT copy them into hy3dpaint.
#
# 0) Clone upstream repo and symlink hy3dpaint into LightX2V:
#    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /path/to/Hunyuan3D-2.1
#    ln -sfn /path/to/Hunyuan3D-2.1/hy3dpaint ${lightx2v_path}/tools/postprocess/hy3dpaint
#    (set hy_repo above to the same clone path for demo assets)
#
# 1) Download RealESRGAN checkpoint (required for texture super-resolution):
#    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
#      -O ${lightx2v_path}/tools/postprocess/hy3dpaint/ckpt/RealESRGAN_x4plus.pth
#
# 2) Build custom_rasterizer CUDA extension:
#    cd ${lightx2v_path}/tools/postprocess/hy3dpaint/custom_rasterizer && pip install -e .
#
# 3) Build DifferentiableRenderer mesh inpaint extension:
#    cd ${lightx2v_path}/tools/postprocess/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh
#
# Requires: pybind11 (`pip install pybind11`) for step 3.

export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=FP16

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
