#!/bin/bash
set -e

# 用法：
#   bash run_hunyuan_image3_t2i_dist_cfg.sh 0,1,2,3
#
# 该脚本是 HunyuanImage3 t2i 的 CFG parallel 入口，结构参考 WAN:
# 普通 t2i 使用 run_hunyuan_image3_t2i.sh；CFG parallel 使用本脚本和 configs/dist_infer 下的 dist cfg。

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"

export lightx2v_path="/data/nvme0/lhd_codes/LightX2V"
export model_path="${model_path:-/data/nvme0/models/tencent/HunyuanImage-3.0}"
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

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "nproc_per_node=${nproc_per_node:-2}"
echo "enable_kv_cache=${enable_kv_cache:-true}"
echo "enable_text_kv_cache=${enable_text_kv_cache:-${enable_kv_cache:-true}}"
echo "use_taylor_cache=${use_taylor_cache:-false}"
echo "hunyuan_cfg_mode=${hunyuan_cfg_mode:-parallel}"
echo "moe_impl=${moe_impl:-flashinfer}"
echo "flashinfer_autotune_mode=${flashinfer_autotune_mode:-tune}"
echo "flashinfer_autotune_cache=${flashinfer_autotune_cache:-${lightx2v_path}/save_results/hunyuan_image3_flashinfer_autotune_t2i_dist_cfg.json}"

source ${lightx2v_path}/scripts/base/base.sh

config_json="${lightx2v_path}/configs/dist_infer/hunyuan_image3_t2i_dist_cfg.json"

torchrun --nproc_per_node="${nproc_per_node:-2}" -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task t2i \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i.png" \
    --seed 42 \
    --hunyuan_cfg_mode "${hunyuan_cfg_mode:-parallel}" \
    --moe_impl "${moe_impl:-flashinfer}" \
    --attn_impl "${attn_impl:-sdpa}" \
    --flashinfer_autotune_mode "${flashinfer_autotune_mode:-tune}" \
    --flashinfer_autotune_cache "${flashinfer_autotune_cache:-${lightx2v_path}/save_results/hunyuan_image3_flashinfer_autotune_t2i.json}" \
    --flashinfer_tune_max_num_tokens "${flashinfer_tune_max_num_tokens:-16384}" \
    --flashinfer_tuning_buckets "${flashinfer_tuning_buckets:-128,256,512,1024,2048,4096,8192,12288,16384}" \
    --flashinfer_autotune_round_up "${flashinfer_autotune_round_up:-true}" \
    --enable_kv_cache "${enable_kv_cache:-true}" \
    --enable_text_kv_cache "${enable_text_kv_cache:-${enable_kv_cache:-true}}" \
    --use_taylor_cache "${use_taylor_cache:-false}" \
    --taylor_cache_interval "${taylor_cache_interval:-5}" \
    --taylor_cache_order "${taylor_cache_order:-2}" \
    --taylor_cache_enable_first_enhance "${taylor_cache_enable_first_enhance:-false}" \
    --taylor_cache_first_enhance_steps "${taylor_cache_first_enhance_steps:-3}" \
    --taylor_cache_enable_tailing_enhance "${taylor_cache_enable_tailing_enhance:-false}" \
    --taylor_cache_tailing_enhance_steps "${taylor_cache_tailing_enhance_steps:-1}" \
    --taylor_cache_low_freqs_order "${taylor_cache_low_freqs_order:-2}" \
    --taylor_cache_high_freqs_order "${taylor_cache_high_freqs_order:-2}"
