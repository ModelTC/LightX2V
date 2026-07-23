#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/hunyuan_image3/run_hunyuan_image3_t2t.sh
#   bash scripts/hunyuan_image3/run_hunyuan_image3_t2t.sh 0,1

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"

export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export model_path="${model_path:-/path/to/HunyuanImage-3-Instruct}"
export HUNYUAN_IMAGE3_REPO_PATH="${HUNYUAN_IMAGE3_REPO_PATH:-/path/to/HunyuanImage-3.0-repo}"
export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    fi
fi

source "${lightx2v_path}/scripts/base/base.sh"

config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2t.json"
prompt="${prompt:-请简要介绍这张图像生成模型的主要能力。}"
save_result_path="${save_result_path:-${lightx2v_path}/save_results/hunyuan_image3_t2t.txt}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "config_json=${config_json}"

python -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task t2t \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "$prompt" \
    --save_result_path "$save_result_path" \
    --seed 42
