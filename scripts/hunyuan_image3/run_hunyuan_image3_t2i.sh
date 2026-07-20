#!/bin/bash
set -e

# 用法：
#   bash run_hunyuan.sh 0
#   bash run_hunyuan.sh 1,2
#   bash run_hunyuan.sh 3,5,6
#
# 如果不传参数，则使用已有的 CUDA_VISIBLE_DEVICES
# 如果也没有设置 CUDA_VISIBLE_DEVICES，则自动使用所有 GPU
# HunyuanImage3 的模型和性能参数统一在 config_json 中设置。

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"

export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export model_path="/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct"
export HUNYUAN_IMAGE3_REPO_PATH="/data/nvme0/lhd_codes/HunyuanImage-3.0"
export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    fi
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

source ${lightx2v_path}/scripts/base/base.sh

config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i.json"
echo "config_json=${config_json}"

python -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task t2i \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i.png" \
    --seed 42
