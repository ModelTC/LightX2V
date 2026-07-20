#!/bin/bash

# Hunyuan3D-2.1 end-to-end inference on MetaX:
# reference image -> shape mesh -> textured mesh.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
default_lightx2v_path=$(cd -- "${script_dir}/../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${default_lightx2v_path}}
hy_repo=${HUNYUAN3D_REPO:-/data/Hunyuan3D-2.1-platform}
model_path=${HUNYUAN3D_MODEL_PATH:-/data/models/Hunyuan3D-2.1}
dino_ckpt_path=${HUNYUAN3D_DINO_PATH:-/data/models/dinov2-giant}
realesrgan_ckpt_path=${HUNYUAN3D_REALESRGAN_PATH:-${model_path}/RealESRGAN_x4plus.pth}
python_bin=${HUNYUAN3D_PYTHON:-/opt/conda/bin/python}
maca_path=${MACA_PATH:-/opt/maca-3.7.1}

if [[ ! -x "${python_bin}" ]]; then
    echo "Missing base Python: ${python_bin}" >&2
    exit 1
fi

if [[ ! -x "${hy_repo}/build_metax.sh" ]]; then
    echo "Missing MetaX build script: ${hy_repo}/build_metax.sh" >&2
    exit 1
fi

if [[ ! -s "${realesrgan_ckpt_path}" ]]; then
    echo "Missing RealESRGAN checkpoint: ${realesrgan_ckpt_path}" >&2
    echo "Place it under the external model directory or set HUNYUAN3D_REALESRGAN_PATH." >&2
    exit 1
fi

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export AI_DEVICE=cuda
export MACA_PATH=${maca_path}
export PATH=$(dirname "${python_bin}"):${MACA_PATH}/bin:${PATH}
torch_lib=$("${python_bin}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH=${torch_lib}:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_DEFAULT_NCHW=${PYTORCH_DEFAULT_NCHW:-1}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-/data/.cache/huggingface/hunyuan3d_metax_modules}
export PYTHONPATH=${hy_repo}/hy3dpaint:${hy_repo}/hy3dpaint/custom_rasterizer:${PYTHONPATH:-}

extension_suffix=$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
rasterizer_extension=${hy_repo}/hy3dpaint/custom_rasterizer/custom_rasterizer_kernel${extension_suffix}
inpaint_extension=${hy_repo}/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor${extension_suffix}

if [[ "${HUNYUAN3D_FORCE_REBUILD:-0}" == "1" || ! -s "${rasterizer_extension}" || ! -s "${inpaint_extension}" ]]; then
    echo "=== Building Hunyuan3D MetaX extensions with the container base environment ==="
    HUNYUAN3D_PYTHON=${python_bin} MACA_PATH=${MACA_PATH} "${hy_repo}/build_metax.sh"
fi

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=FP16

mkdir -p "${lightx2v_path}/save_results/hunyuan3d-metax" "${HF_MODULES_CACHE}"

echo "=== Step 1/2: shape generation on MetaX ==="
"${python_bin}" -m lightx2v.infer \
    --model_cls hunyuan3d \
    --task i23d \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/metax/hunyuan3d_shape.json" \
    --image_path "${hy_repo}/assets/demo.png" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan3d-metax/shape.glb" \
    --seed 42

test -s "${lightx2v_path}/save_results/hunyuan3d-metax/shape.glb"
echo "Saved mesh: ${lightx2v_path}/save_results/hunyuan3d-metax/shape.glb"

paint_args=()
if [[ "${HUNYUAN3D_NO_REMESH:-0}" == "1" ]]; then
    paint_args+=(--no_remesh)
fi

echo "=== Step 2/2: mesh texture generation on MetaX ==="
"${python_bin}" "${lightx2v_path}/tools/postprocess/postprocess_paint.py" \
    --hy_repo "${hy_repo}" \
    --model_path "${model_path}" \
    --mesh_path "${lightx2v_path}/save_results/hunyuan3d-metax/shape.glb" \
    --image_path "${hy_repo}/assets/demo.png" \
    --save_path "${lightx2v_path}/save_results/hunyuan3d-metax/textured.glb" \
    --device cuda \
    --dino_ckpt_path "${dino_ckpt_path}" \
    --realesrgan_ckpt_path "${realesrgan_ckpt_path}" \
    --render_size 2048 \
    --texture_size 4096 \
    --max_num_view 6 \
    --resolution 512 \
    "${paint_args[@]}"

test -s "${lightx2v_path}/save_results/hunyuan3d-metax/textured.glb"
echo "Saved textured mesh: ${lightx2v_path}/save_results/hunyuan3d-metax/textured.glb"
echo "All outputs in: ${lightx2v_path}/save_results/hunyuan3d-metax"
