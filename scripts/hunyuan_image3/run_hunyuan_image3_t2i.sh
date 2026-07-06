#!/bin/bash
set -e

# 用法：
#   bash run_hunyuan.sh 0
#   bash run_hunyuan.sh 1,2
#   bash run_hunyuan.sh 3,5,6
#
# 如果不传参数，则使用已有的 CUDA_VISIBLE_DEVICES
# 如果也没有设置 CUDA_VISIBLE_DEVICES，则自动使用所有 GPU
# Cache 开关直接写在 python 入口参数里，测试时改入口处的 true/false/数字即可。

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"

export lightx2v_path="/data/nvme0/lhd_codes/LightX2V"
export model_path="/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct"
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
echo "enable_kv_cache=${enable_kv_cache:-true}"
echo "enable_text_kv_cache=${enable_text_kv_cache:-${enable_kv_cache:-true}}"
echo "use_taylor_cache=${use_taylor_cache:-false}"
echo "use_meanflow=${use_meanflow:-false}"
echo "guidance_rescale=${guidance_rescale:-0.0}"
echo "enable_auto_image_size=${enable_auto_image_size:-false}"

source ${lightx2v_path}/scripts/base/base.sh

config_json="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_t2i.json"

python -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task t2i \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗" \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_image3_t2i.png" \
    --seed 42 \
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
    --taylor_cache_high_freqs_order "${taylor_cache_high_freqs_order:-2}" \
    --use_meanflow "${use_meanflow:-false}" \
    --guidance_rescale "${guidance_rescale:-0.0}" \
    --enable_auto_image_size "${enable_auto_image_size:-false}" \
    --hunyuan_image_size "${hunyuan_image_size:-}" \
    --hunyuan_timesteps "${hunyuan_timesteps:-}" \
    --hunyuan_sigmas "${hunyuan_sigmas:-}" \
    --attn_impl "${attn_impl:-flash_attention_2}" \
    --moe_impl "${moe_impl:-eager}" \
    --rewrite "${rewrite:-false}" \
    --sys_deepseek_prompt "${sys_deepseek_prompt:-universal}" \
    --reproduce "${reproduce:-false}"
