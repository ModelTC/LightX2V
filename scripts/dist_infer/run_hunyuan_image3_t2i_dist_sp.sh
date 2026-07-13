#!/bin/bash
set -euo pipefail

# Generic HunyuanImage3 T2I sequence-parallel launcher.
# Prefer the backend-specific wrappers next to this file for normal use.

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"
SP_SIZE="${SP_SIZE:-2}"
SP_ATTN_TYPE="${SP_ATTN_TYPE:-ulysses}"
SP_PIPELINE_LAYOUT="${SP_PIPELINE_LAYOUT:-interleaved}"

export lightx2v_path="${lightx2v_path:-/data/nvme0/lhd_codes/LightX2V}"
export model_path="${model_path:-/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct}"
export HUNYUAN_IMAGE3_REPO_PATH="${HUNYUAN_IMAGE3_REPO_PATH:-/data/nvme0/lhd_codes/HunyuanImage-3.0}"
export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    fi
fi

if ! [[ "$SP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "SP_SIZE must be a positive integer, got: $SP_SIZE" >&2
    exit 2
fi
if [ "$SP_ATTN_TYPE" != "kv_all_gather" ] && [ "$SP_ATTN_TYPE" != "ulysses" ]; then
    echo "SP_ATTN_TYPE must be kv_all_gather or ulysses, got: $SP_ATTN_TYPE" >&2
    exit 2
fi
if [ "$SP_PIPELINE_LAYOUT" != "interleaved" ] && [ "$SP_PIPELINE_LAYOUT" != "contiguous" ]; then
    echo "SP_PIPELINE_LAYOUT must be interleaved or contiguous, got: $SP_PIPELINE_LAYOUT" >&2
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

if [ "$SP_ATTN_TYPE" = "ulysses" ]; then
    config_json="${lightx2v_path}/configs/dist_infer/hunyuan_image3_t2i_dist_ulysses.json"
else
    config_json="${lightx2v_path}/configs/dist_infer/hunyuan_image3_t2i_dist_kv_all_gather.json"
fi
default_autotune_cache="${lightx2v_path}/save_results/hunyuan_image3_flashinfer_${SP_ATTN_TYPE}_sp${SP_SIZE}_t2i.json"
resolved_autotune_cache="${flashinfer_autotune_cache:-$default_autotune_cache}"
if [ -n "${flashinfer_autotune_mode:-}" ]; then
    resolved_autotune_mode="$flashinfer_autotune_mode"
elif [ "${flashinfer_autotune:-true}" = "false" ] || [ "${flashinfer_autotune:-true}" = "0" ]; then
    resolved_autotune_mode="off"
else
    resolved_autotune_mode="tune"
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SP_SIZE=${SP_SIZE}, SP_ATTN_TYPE=${SP_ATTN_TYPE}, layout=${SP_PIPELINE_LAYOUT}"
for ((rank=0; rank<SP_SIZE; rank++)); do
    lane=()
    if [ "$SP_PIPELINE_LAYOUT" = "interleaved" ]; then
        for ((idx=rank; idx<visible_gpu_count; idx+=SP_SIZE)); do lane+=("${visible_gpu_ids[$idx]}"); done
    else
        start=$((rank * pipeline_gpus_per_lane))
        for ((idx=start; idx<start+pipeline_gpus_per_lane; idx++)); do lane+=("${visible_gpu_ids[$idx]}"); done
    fi
    lane_csv=$(IFS=,; echo "${lane[*]}")
    echo "SP rank ${rank} pipeline lane: ${lane_csv}"
done
echo "hunyuan_cfg_mode=serial, moe_impl=${moe_impl:-flashinfer}, attn_impl=${attn_impl:-flash_attention_2}"
echo "enable_kv_cache=${enable_kv_cache:-true}, flashinfer_autotune_mode=${resolved_autotune_mode}"
echo "flashinfer_autotune_cache=${resolved_autotune_cache}"

source "${lightx2v_path}/scripts/base/base.sh"
export HUNYUAN_IMAGE3_PIPELINE_LAYOUT="$SP_PIPELINE_LAYOUT"

log_root="${HUNYUAN_SP_LOG_DIR:-${lightx2v_path}/save_results/torchrun_logs}"
mkdir -p "$log_root"

torchrun \
    --standalone \
    --nproc_per_node="$SP_SIZE" \
    --max_restarts=0 \
    --log-dir "$log_root" \
    --tee 3 \
    "${lightx2v_path}/scripts/dist_infer/hunyuan_image3_sp_entry.py" \
    --model_cls hunyuan_image3 \
    --task t2i \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i_${SP_ATTN_TYPE}_sp${SP_SIZE}.png" \
    --seed 42 \
    --hunyuan_sp_size "$SP_SIZE" \
    --hunyuan_sp_attn_type "$SP_ATTN_TYPE" \
    --hunyuan_pipeline_layout "$SP_PIPELINE_LAYOUT" \
    --hunyuan_cfg_mode serial \
    --moe_impl "${moe_impl:-flashinfer}" \
    --attn_impl "${attn_impl:-flash_attention_2}" \
    --flashinfer_autotune_mode "$resolved_autotune_mode" \
    --flashinfer_autotune_cache "$resolved_autotune_cache" \
    --flashinfer_tune_max_num_tokens "${flashinfer_tune_max_num_tokens:-16384}" \
    --flashinfer_tuning_buckets "${flashinfer_tuning_buckets:-128,256,512,1024,2048,4096,8192,12288,16384}" \
    --flashinfer_autotune_round_up "${flashinfer_autotune_round_up:-true}" \
    --enable_kv_cache "${enable_kv_cache:-true}" \
    --enable_text_kv_cache "${enable_text_kv_cache:-${enable_kv_cache:-true}}" \
    --use_taylor_cache false
