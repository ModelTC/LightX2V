#!/bin/bash
set -e

# 用法：
#   bash run_hunyuan_image3_t2i_dist_cfg.sh 0,1,2,3
#
# 该脚本是 HunyuanImage3 t2i 的 CFG parallel 入口，结构参考 WAN:
# 普通 t2i 使用 run_hunyuan_image3_t2i.sh；CFG parallel 使用本脚本和 configs/dist_infer 下的 dist cfg。

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"

export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export model_path="${model_path:-/path/to/HunyuanImage-3.0-model}"
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

config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i_dist_cfg.json"
nproc_per_node=$(python - "$config_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    parallel = json.load(f)["parallel"]
print(int(parallel.get("cfg_p_size", 1)) * int(parallel.get("seq_p_size", 1)))
PY
)
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "nproc_per_node=${nproc_per_node}"
echo "config_json=${config_json}"

source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node="${nproc_per_node}" -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task t2i \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i_cfg_parallel.png" \
    --seed 42
