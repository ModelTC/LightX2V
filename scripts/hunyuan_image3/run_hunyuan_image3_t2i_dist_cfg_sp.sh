#!/bin/bash
set -euo pipefail

# Common HunyuanImage3 T2I launcher for CFG parallel + sequence parallel.
# A logical torchrun rank owns one interleaved pipeline lane:
#   world_size = CFG_SIZE * SP_SIZE
#   pipeline_size = visible_gpu_count / world_size

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"
SP_ATTN_TYPE="${SP_ATTN_TYPE:-ulysses}"

export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export model_path="/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct"
export HUNYUAN_IMAGE3_REPO_PATH="/data/nvme0/lhd_codes/HunyuanImage-3.0"
export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"

if [ "$SP_ATTN_TYPE" = "ulysses" ]; then
    config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i_dist_cfg_ulysses.json"
elif [ "$SP_ATTN_TYPE" = "kv_all_gather" ]; then
    config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i_dist_cfg_kv_all_gather.json"
else
    echo "SP_ATTN_TYPE must select either the ulysses or kv_all_gather JSON config, got: $SP_ATTN_TYPE" >&2
    exit 2
fi

read -r CFG_SIZE SP_SIZE configured_sp_attn_type < <(python - "$config_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    parallel = json.load(f)["parallel"]
print(parallel["cfg_p_size"], parallel["seq_p_size"], parallel["seq_p_attn_type"])
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

if [ "$CFG_SIZE" != "2" ]; then
    echo "HunyuanImage3 CFG parallel has exactly two cond/uncond branches; parallel.cfg_p_size in $config_json must be 2, got: $CFG_SIZE" >&2
    exit 2
fi
if ! [[ "$SP_SIZE" =~ ^[1-9][0-9]*$ ]] || [ "$SP_SIZE" -le 1 ]; then
    echo "Hybrid CFG+SP requires parallel.seq_p_size in $config_json to be an integer greater than 1, got: $SP_SIZE" >&2
    exit 2
fi

IFS=',' read -r -a raw_visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-}"
visible_gpu_ids=()
declare -A seen_gpu_ids=()
for gpu_id in "${raw_visible_gpu_ids[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [ -z "$gpu_id" ]; then
        continue
    fi
    if [ -n "${seen_gpu_ids[$gpu_id]:-}" ]; then
        echo "Duplicate GPU id in CUDA_VISIBLE_DEVICES: $gpu_id" >&2
        exit 2
    fi
    seen_gpu_ids[$gpu_id]=1
    visible_gpu_ids+=("$gpu_id")
done
visible_gpu_count=${#visible_gpu_ids[@]}
if [ "$visible_gpu_count" -eq 0 ]; then
    echo "No visible CUDA devices were found." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${visible_gpu_ids[*]}")

world_size=$((CFG_SIZE * SP_SIZE))
if [ $((visible_gpu_count % world_size)) -ne 0 ]; then
    echo "Visible GPU count ($visible_gpu_count) must be divisible by CFG_SIZE * SP_SIZE ($CFG_SIZE * $SP_SIZE = $world_size)." >&2
    exit 2
fi
if [ "$SP_ATTN_TYPE" = "ulysses" ] && { [ $((32 % SP_SIZE)) -ne 0 ] || [ $((8 % SP_SIZE)) -ne 0 ]; }; then
    echo "Ulysses SP_SIZE must divide both 32 Q heads and 8 KV heads, got: $SP_SIZE" >&2
    exit 2
fi

pipeline_gpus_per_lane=$((visible_gpu_count / world_size))
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CFG_SIZE=${CFG_SIZE}, SP_SIZE=${SP_SIZE}, SP_ATTN_TYPE=${SP_ATTN_TYPE}, WORLD_SIZE=${world_size}"
echo "pipeline_gpus_per_lane=${pipeline_gpus_per_lane}, layout=interleaved"
for ((rank=0; rank<world_size; rank++)); do
    cfg_rank=$((rank / SP_SIZE))
    seq_rank=$((rank % SP_SIZE))
    lane=()
    for ((idx=rank; idx<visible_gpu_count; idx+=world_size)); do
        lane+=("${visible_gpu_ids[$idx]}")
    done
    lane_csv=$(IFS=,; echo "${lane[*]}")
    echo "global rank ${rank} (cfg=${cfg_rank}, seq=${seq_rank}) pipeline lane: ${lane_csv}"
done
echo "config_json=${config_json}"

source "${lightx2v_path}/scripts/base/base.sh"

log_root="${HUNYUAN_CFG_SP_LOG_DIR:-${lightx2v_path}/save_results/torchrun_logs}"
mkdir -p "$log_root"

torchrun \
    --standalone \
    --nproc_per_node="$world_size" \
    --max_restarts=0 \
    --log-dir "$log_root" \
    --tee 3 \
    -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task t2i \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i_cfg${CFG_SIZE}_${SP_ATTN_TYPE}_sp${SP_SIZE}_pp${pipeline_gpus_per_lane}.png" \
    --seed 42
