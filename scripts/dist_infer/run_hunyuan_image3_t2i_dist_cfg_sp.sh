#!/bin/bash
set -euo pipefail

# Common HunyuanImage3 T2I launcher for CFG parallel + sequence parallel.
# A logical torchrun rank owns one interleaved pipeline lane:
#   world_size = CFG_SIZE * SP_SIZE
#   pipeline_size = visible_gpu_count / world_size

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"
CFG_SIZE="${CFG_SIZE:-2}"
SP_SIZE="${SP_SIZE:-2}"
SP_ATTN_TYPE="${SP_ATTN_TYPE:-ulysses}"

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

if [ "$CFG_SIZE" != "2" ]; then
    echo "HunyuanImage3 CFG parallel has exactly two cond/uncond branches; CFG_SIZE must be 2, got: $CFG_SIZE" >&2
    exit 2
fi
if ! [[ "$SP_SIZE" =~ ^[1-9][0-9]*$ ]] || [ "$SP_SIZE" -le 1 ]; then
    echo "Hybrid CFG+SP requires SP_SIZE to be an integer greater than 1, got: $SP_SIZE" >&2
    exit 2
fi
if [ "$SP_ATTN_TYPE" != "kv_all_gather" ] && [ "$SP_ATTN_TYPE" != "ulysses" ]; then
    echo "SP_ATTN_TYPE must be kv_all_gather or ulysses, got: $SP_ATTN_TYPE" >&2
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
if [ "$SP_ATTN_TYPE" = "ulysses" ]; then
    config_json="${lightx2v_path}/configs/dist_infer/hunyuan_image3_t2i_dist_cfg_ulysses.json"
    default_attn_impl="sdpa"
else
    config_json="${lightx2v_path}/configs/dist_infer/hunyuan_image3_t2i_dist_cfg_kv_all_gather.json"
    default_attn_impl="flash_attention_2"
fi
resolved_attn_impl="${attn_impl:-$default_attn_impl}"
default_autotune_cache="${lightx2v_path}/save_results/hunyuan_image3_flashinfer_${SP_ATTN_TYPE}_cfg${CFG_SIZE}_sp${SP_SIZE}_pp${pipeline_gpus_per_lane}_t2i.json"
resolved_autotune_cache="${flashinfer_autotune_cache:-$default_autotune_cache}"
if [ -n "${flashinfer_autotune_mode:-}" ]; then
    resolved_autotune_mode="$flashinfer_autotune_mode"
elif [ "${flashinfer_autotune:-true}" = "false" ] || [ "${flashinfer_autotune:-true}" = "0" ]; then
    resolved_autotune_mode="off"
else
    resolved_autotune_mode="tune"
fi

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
echo "hunyuan_cfg_mode=parallel, moe_impl=${moe_impl:-flashinfer}, attn_impl=${resolved_attn_impl}"
echo "enable_kv_cache=${enable_kv_cache:-true}, enable_text_kv_cache=${enable_text_kv_cache:-${enable_kv_cache:-true}}"
echo "flashinfer_autotune_mode=${resolved_autotune_mode}, flashinfer_autotune_cache=${resolved_autotune_cache}"

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
    --seed 42 \
    --hunyuan_sp_size "$SP_SIZE" \
    --hunyuan_sp_attn_type "$SP_ATTN_TYPE" \
    --hunyuan_cfg_mode parallel \
    --moe_impl "${moe_impl:-flashinfer}" \
    --attn_impl "$resolved_attn_impl" \
    --flashinfer_autotune_mode "$resolved_autotune_mode" \
    --flashinfer_autotune_cache "$resolved_autotune_cache" \
    --flashinfer_tune_max_num_tokens "${flashinfer_tune_max_num_tokens:-16384}" \
    --flashinfer_tuning_buckets "${flashinfer_tuning_buckets:-128,256,512,1024,2048,4096,8192,12288,16384}" \
    --flashinfer_autotune_round_up "${flashinfer_autotune_round_up:-true}" \
    --enable_kv_cache "${enable_kv_cache:-true}" \
    --enable_text_kv_cache "${enable_text_kv_cache:-${enable_kv_cache:-true}}" \
    --use_taylor_cache false
