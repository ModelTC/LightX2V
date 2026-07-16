#!/bin/bash

set -euo pipefail

lightx2v_path=${LIGHTX2V_PATH:-/data/wushuo1/LightX2V}
hy_repo=${HUNYUAN3D_REPO:-/data/wushuo1/Hunyuan3D-2.1-platform}
model_path=${HUNYUAN3D_MODEL_PATH:-/data/wushuo1/models/Hunyuan3D-2.1}
python_bin=${HUNYUAN3D_PYTHON:-/data/wushuo1/envs/hunyuan3d-ascend/bin/python}

if [[ ! -s "${hy_repo}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]]; then
    echo "Missing RealESRGAN checkpoint: ${hy_repo}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" >&2
    echo "Download it as documented in ${hy_repo}/README.md." >&2
    exit 1
fi

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0
export AI_DEVICE=npu
export HF_MODULES_CACHE=/data/wushuo1/cache/huggingface/hunyuan3d_ascend_modules
export PYTHONPATH=${PYTHONPATH:-}

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=FP16
export PROFILING_DEBUG_LEVEL=1

mkdir -p \
    "${lightx2v_path}/save_results/hunyuan3d" \
    /data/wushuo1/cache/hunyuan3d/shape \
    "${HF_MODULES_CACHE}"

echo "=== Step 1/2: shape generation ==="
cd /data/wushuo1/cache/hunyuan3d/shape
"${python_bin}" -m lightx2v.infer \
    --model_cls hunyuan3d \
    --task i23d \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/ascend_npu/hunyuan3d_shape.json" \
    --image_path "${hy_repo}/assets/demo.png" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan3d/e2e_shape.glb" \
    --seed 42

test -s "${lightx2v_path}/save_results/hunyuan3d/e2e_shape.glb"
echo "Saved mesh: ${lightx2v_path}/save_results/hunyuan3d/e2e_shape.glb"

extra_args=()
if [[ "${HUNYUAN3D_NO_REMESH:-0}" == "1" ]]; then
    extra_args+=(--no_remesh)
fi

echo "=== Step 2/2: mesh texture (paint) ==="
"${python_bin}" "${lightx2v_path}/tools/postprocess/postprocess_paint.py" \
    --hy_repo "${hy_repo}" \
    --model_path "${model_path}" \
    --mesh_path "${lightx2v_path}/save_results/hunyuan3d/e2e_shape.glb" \
    --image_path "${hy_repo}/assets/demo.png" \
    --save_path "${lightx2v_path}/save_results/hunyuan3d/e2e_textured.glb" \
    --device npu \
    --dino_ckpt_path /data/wushuo1/models/dinov2-giant \
    --realesrgan_ckpt_path "${hy_repo}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" \
    --render_size 2048 \
    --texture_size 4096 \
    --max_num_view 6 \
    --resolution 512 \
    "${extra_args[@]}"

test -s "${lightx2v_path}/save_results/hunyuan3d/e2e_textured.glb"
echo "Saved textured mesh: ${lightx2v_path}/save_results/hunyuan3d/e2e_textured.glb"
echo "All outputs in: ${lightx2v_path}/save_results/hunyuan3d"
