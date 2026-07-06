#!/bin/bash

export lightx2v_path=${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export model_path=${model_path:-/path/to/HunyuanImage-3-Instruct}
export image_path=${image_path:-/path/to/input.png}
export ENABLE_KV_CACHE="${ENABLE_KV_CACHE:-true}"
export ENABLE_TEXT_KV_CACHE="${ENABLE_TEXT_KV_CACHE:-$ENABLE_KV_CACHE}"
export USE_TAYLOR_CACHE="${USE_TAYLOR_CACHE:-false}"
export TAYLOR_CACHE_INTERVAL="${TAYLOR_CACHE_INTERVAL:-5}"
export TAYLOR_CACHE_ORDER="${TAYLOR_CACHE_ORDER:-2}"
export TAYLOR_CACHE_ENABLE_FIRST_ENHANCE="${TAYLOR_CACHE_ENABLE_FIRST_ENHANCE:-false}"
export TAYLOR_CACHE_FIRST_ENHANCE_STEPS="${TAYLOR_CACHE_FIRST_ENHANCE_STEPS:-3}"
export TAYLOR_CACHE_ENABLE_TAILING_ENHANCE="${TAYLOR_CACHE_ENABLE_TAILING_ENHANCE:-false}"
export TAYLOR_CACHE_TAILING_ENHANCE_STEPS="${TAYLOR_CACHE_TAILING_ENHANCE_STEPS:-1}"
export TAYLOR_CACHE_LOW_FREQS_ORDER="${TAYLOR_CACHE_LOW_FREQS_ORDER:-2}"
export TAYLOR_CACHE_HIGH_FREQS_ORDER="${TAYLOR_CACHE_HIGH_FREQS_ORDER:-2}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))
    fi
fi

source ${lightx2v_path}/scripts/base/base.sh

echo "ENABLE_KV_CACHE=${ENABLE_KV_CACHE}"
echo "ENABLE_TEXT_KV_CACHE=${ENABLE_TEXT_KV_CACHE}"
echo "USE_TAYLOR_CACHE=${USE_TAYLOR_CACHE}"
echo "USE_MEANFLOW=${USE_MEANFLOW:-false}"
echo "GUIDANCE_RESCALE=${GUIDANCE_RESCALE:-0.0}"
echo "ENABLE_AUTO_IMAGE_SIZE=${ENABLE_AUTO_IMAGE_SIZE:-false}"

base_config="${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_i2i.json"
runtime_config="$(mktemp /tmp/hunyuan_image3_i2i_config.XXXXXX.json)"
trap 'rm -f "$runtime_config"' EXIT

python - "$base_config" "$runtime_config" <<'PY'
import json
import os
import sys

def env_bool(name, default=False):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")

def env_int(name, default):
    return int(os.environ.get(name, default))

base_config, runtime_config = sys.argv[1:3]
with open(base_config, "r") as f:
    config = json.load(f)
config["enable_kv_cache"] = env_bool("ENABLE_KV_CACHE", True)
config["enable_text_kv_cache"] = env_bool("ENABLE_TEXT_KV_CACHE", config["enable_kv_cache"])
config["use_taylor_cache"] = env_bool("USE_TAYLOR_CACHE", False)
config["taylor_cache_interval"] = env_int("TAYLOR_CACHE_INTERVAL", 5)
config["taylor_cache_order"] = env_int("TAYLOR_CACHE_ORDER", 2)
config["taylor_cache_enable_first_enhance"] = env_bool("TAYLOR_CACHE_ENABLE_FIRST_ENHANCE", False)
config["taylor_cache_first_enhance_steps"] = env_int("TAYLOR_CACHE_FIRST_ENHANCE_STEPS", 3)
config["taylor_cache_enable_tailing_enhance"] = env_bool("TAYLOR_CACHE_ENABLE_TAILING_ENHANCE", False)
config["taylor_cache_tailing_enhance_steps"] = env_int("TAYLOR_CACHE_TAILING_ENHANCE_STEPS", 1)
config["taylor_cache_low_freqs_order"] = env_int("TAYLOR_CACHE_LOW_FREQS_ORDER", 2)
config["taylor_cache_high_freqs_order"] = env_int("TAYLOR_CACHE_HIGH_FREQS_ORDER", 2)
with open(runtime_config, "w") as f:
    json.dump(config, f, indent=4)
PY

python -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task i2i \
    --model_path $model_path \
    --config_json "$runtime_config" \
    --prompt "新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。 鱼眼镜头，背景是房间门口，近景，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清，绒毛细节，面部特写。" \
    --image_path "${image_path}" \
    --save_result_path ${lightx2v_path}/save_results/hunyuan_image3_i2i.png \
    --seed 42 \
    --enable_kv_cache "${ENABLE_KV_CACHE:-true}" \
    --enable_text_kv_cache "${ENABLE_TEXT_KV_CACHE:-${ENABLE_KV_CACHE:-true}}" \
    --use_taylor_cache "${USE_TAYLOR_CACHE:-false}" \
    --taylor_cache_interval "${TAYLOR_CACHE_INTERVAL:-5}" \
    --taylor_cache_order "${TAYLOR_CACHE_ORDER:-2}" \
    --taylor_cache_enable_first_enhance "${TAYLOR_CACHE_ENABLE_FIRST_ENHANCE:-false}" \
    --taylor_cache_first_enhance_steps "${TAYLOR_CACHE_FIRST_ENHANCE_STEPS:-3}" \
    --taylor_cache_enable_tailing_enhance "${TAYLOR_CACHE_ENABLE_TAILING_ENHANCE:-false}" \
    --taylor_cache_tailing_enhance_steps "${TAYLOR_CACHE_TAILING_ENHANCE_STEPS:-1}" \
    --taylor_cache_low_freqs_order "${TAYLOR_CACHE_LOW_FREQS_ORDER:-2}" \
    --taylor_cache_high_freqs_order "${TAYLOR_CACHE_HIGH_FREQS_ORDER:-2}" \
    --use_meanflow "${USE_MEANFLOW:-false}" \
    --guidance_rescale "${GUIDANCE_RESCALE:-0.0}" \
    --enable_auto_image_size "${ENABLE_AUTO_IMAGE_SIZE:-false}" \
    --hunyuan_image_size "${HUNYUAN_IMAGE_SIZE:-}" \
    --hunyuan_timesteps "${HUNYUAN_TIMESTEPS:-}" \
    --hunyuan_sigmas "${HUNYUAN_SIGMAS:-}" \
    --attn_impl "${ATTN_IMPL:-sdpa}" \
    --moe_impl "${MOE_IMPL:-eager}" \
    --rewrite "${REWRITE:-false}" \
    --sys_deepseek_prompt "${SYS_DEEPSEEK_PROMPT:-universal}" \
    --reproduce "${REPRODUCE:-false}"
