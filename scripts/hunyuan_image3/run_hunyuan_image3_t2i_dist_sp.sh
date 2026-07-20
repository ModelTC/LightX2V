#!/bin/bash
set -euo pipefail

# Generic HunyuanImage3 T2I sequence-parallel launcher.
# Prefer the backend-specific wrappers next to this file for normal use.

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"
SP_ATTN_TYPE="${SP_ATTN_TYPE:-ulysses}"

export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export model_path="${model_path:-/path/to/HunyuanImage-3.0-model}"
export HUNYUAN_IMAGE3_REPO_PATH="${HUNYUAN_IMAGE3_REPO_PATH:-/path/to/HunyuanImage-3.0-repo}"
export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"

if [ "$SP_ATTN_TYPE" = "ulysses" ]; then
    config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i_dist_ulysses.json"
elif [ "$SP_ATTN_TYPE" = "kv_all_gather" ]; then
    config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i_dist_kv_all_gather.json"
else
    echo "SP_ATTN_TYPE must select either the ulysses or kv_all_gather JSON config, got: $SP_ATTN_TYPE" >&2
    exit 2
fi

read -r SP_SIZE configured_sp_attn_type < <(python - "$config_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    parallel = json.load(f)["parallel"]
print(parallel["seq_p_size"], parallel["seq_p_attn_type"])
PY
)
if [ "$SP_ATTN_TYPE" != "$configured_sp_attn_type" ]; then
    echo "Selected SP backend ($SP_ATTN_TYPE) does not match config parallel.seq_p_attn_type ($configured_sp_attn_type)." >&2
    exit 2
fi

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    fi
fi

if ! [[ "$SP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "parallel.seq_p_size in $config_json must be a positive integer, got: $SP_SIZE" >&2
    exit 2
fi
IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-}"
visible_gpu_count=${#visible_gpu_ids[@]}
if [ "$visible_gpu_count" -eq 0 ] || [ -z "${visible_gpu_ids[0]}" ]; then
    echo "No visible CUDA devices were found." >&2
    exit 2
fi
declare -A seen_gpu_ids=()
for gpu_id in "${visible_gpu_ids[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [ -n "${seen_gpu_ids[$gpu_id]:-}" ]; then
        echo "Duplicate GPU id in CUDA_VISIBLE_DEVICES: $gpu_id" >&2
        exit 2
    fi
    seen_gpu_ids[$gpu_id]=1
done
if [ $((visible_gpu_count % SP_SIZE)) -ne 0 ]; then
    echo "Visible GPU count ($visible_gpu_count) must be divisible by SP_SIZE ($SP_SIZE)." >&2
    exit 2
fi
if [ "$SP_ATTN_TYPE" = "ulysses" ] && { [ $((32 % SP_SIZE)) -ne 0 ] || [ $((8 % SP_SIZE)) -ne 0 ]; }; then
    echo "Ulysses SP_SIZE must divide both 32 Q heads and 8 KV heads, got: $SP_SIZE" >&2
    exit 2
fi

pipeline_gpus_per_lane=$((visible_gpu_count / SP_SIZE))
min_pipeline_gpus_per_lane="${MIN_PIPELINE_GPUS_PER_LANE:-3}"
if [ "$pipeline_gpus_per_lane" -lt "$min_pipeline_gpus_per_lane" ] && [ "${ALLOW_UNSAFE_SP_MEMORY_LAYOUT:-false}" != "true" ]; then
    echo "Each SP lane has only ${pipeline_gpus_per_lane} GPU(s); this BF16 checkpoint expects at least ${min_pipeline_gpus_per_lane}." >&2
    echo "Set ALLOW_UNSAFE_SP_MEMORY_LAYOUT=true only for a separately validated quantized/offload setup." >&2
    exit 2
fi

min_free_gpu_memory_mib="${MIN_FREE_GPU_MEMORY_MIB:-50000}"
if command -v nvidia-smi >/dev/null 2>&1; then
    for gpu_id in "${visible_gpu_ids[@]}"; do
        free_mib=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)
        if [[ "$free_mib" =~ ^[0-9]+$ ]]; then
            echo "GPU ${gpu_id}: free_memory=${free_mib} MiB"
            if [ "$free_mib" -lt "$min_free_gpu_memory_mib" ] && [ "${ALLOW_LOW_FREE_MEMORY:-false}" != "true" ]; then
                echo "GPU ${gpu_id} has only ${free_mib} MiB free; require at least ${min_free_gpu_memory_mib} MiB for this launcher." >&2
                echo "Choose an idle GPU or set ALLOW_LOW_FREE_MEMORY=true after validating memory manually." >&2
                exit 2
            fi
        fi
    done
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SP_SIZE=${SP_SIZE}, SP_ATTN_TYPE=${SP_ATTN_TYPE}, layout=interleaved"
for ((rank=0; rank<SP_SIZE; rank++)); do
    lane=()
    for ((idx=rank; idx<visible_gpu_count; idx+=SP_SIZE)); do lane+=("${visible_gpu_ids[$idx]}"); done
    lane_csv=$(IFS=,; echo "${lane[*]}")
    echo "SP rank ${rank} pipeline lane: ${lane_csv}"
done
echo "config_json=${config_json}"

source "${lightx2v_path}/scripts/base/base.sh"

log_root="${HUNYUAN_SP_LOG_DIR:-${lightx2v_path}/save_results/torchrun_logs}"
mkdir -p "$log_root"

torchrun \
    --standalone \
    --nproc_per_node="$SP_SIZE" \
    --max_restarts=0 \
    --log-dir "$log_root" \
    --tee 3 \
    -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task t2i \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i_${SP_ATTN_TYPE}_sp${SP_SIZE}.png" \
    --seed 42
