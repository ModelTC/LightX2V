#!/usr/bin/env bash

export LIGHTX2V_PATH="${LIGHTX2V_PATH:-/data/LightX2V-mlu}"
export HY_REPO="${HY_REPO:-/data/Hunyuan3D-2.1-platform-mlu}"
export MODEL_PATH="${MODEL_PATH:-/data/models/Hunyuan3D-2.1}"
export DINO_PATH="${DINO_PATH:-/data/models/dinov2-giant}"
export HY_ENV="${HY_ENV:-/torch/venv3/hunyuan3d-mlu}"
export HY_PY="${HY_PY:-${HY_ENV}/bin/python}"

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export PATH="${NEUWARE_HOME}/bin:${HY_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib:/torch/venv3/pytorch_infer/lib/python3.10/site-packages/torch/lib:/torch/venv3/pytorch_infer/lib/python3.10/site-packages/torch_mlu/csrc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

unset CN_VISIBLE_DEVICES
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0}"
export AI_DEVICE="mlu:0"
export PLATFORM="cambricon_mlu"
export TORCH_BANG_ARCH_LIST="5.0"

export PYTHONPATH="${LIGHTX2V_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export DTYPE="FP16"
export SENSITIVE_LAYER_DTYPE="None"
export PYTORCH_MLU_ALLOC_CONF="${PYTORCH_MLU_ALLOC_CONF:-expandable_segments:True}"

# Sourcing only prepares the build/runtime environment. Executing starts the
# complete Shape + Paint pipeline.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

set -euo pipefail

mkdir -p "${LIGHTX2V_PATH}/save_results/hunyuan3d-mlu"

echo "=== Step 1/2: Shape on ${AI_DEVICE} ==="
"${HY_PY}" -m lightx2v.infer \
    --model_cls hunyuan3d \
    --task i23d \
    --model_path "${MODEL_PATH}" \
    --config_json "${LIGHTX2V_PATH}/configs/platforms/mlu/hunyuan3d_shape_mlu.json" \
    --image_path "${HY_REPO}/assets/demo.png" \
    --save_result_path "${LIGHTX2V_PATH}/save_results/hunyuan3d-mlu/shape.glb" \
    --seed 42

echo "=== Step 2/2: Paint on ${AI_DEVICE} ==="
"${HY_PY}" -u "${LIGHTX2V_PATH}/tools/postprocess/postprocess_paint.py" \
    --model_path "${MODEL_PATH}" \
    --hy_repo "${HY_REPO}" \
    --mesh_path "${LIGHTX2V_PATH}/save_results/hunyuan3d-mlu/shape.glb" \
    --image_path "${HY_REPO}/assets/demo.png" \
    --save_path "${LIGHTX2V_PATH}/save_results/hunyuan3d-mlu/textured.glb" \
    --max_num_view 6 \
    --resolution 512 \
    --device "${AI_DEVICE}" \
    --dino_ckpt_path "${DINO_PATH}" \
    --render_size 2048 \
    --texture_size 4096 \
    --use_remesh \
    --save_glb

echo "Shape: ${LIGHTX2V_PATH}/save_results/hunyuan3d-mlu/shape.glb"
echo "Textured mesh: ${LIGHTX2V_PATH}/save_results/hunyuan3d-mlu/textured.glb"
