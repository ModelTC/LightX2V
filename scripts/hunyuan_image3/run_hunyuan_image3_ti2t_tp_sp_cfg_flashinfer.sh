#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/hunyuan_image3/run_hunyuan_image3_ti2t_tp_sp_cfg_flashinfer.sh
#   bash scripts/hunyuan_image3/run_hunyuan_image3_ti2t_tp_sp_cfg_flashinfer.sh 0,1,2,3
#
# TI2T has no diffusion CFG branch. The cfg_p_size=1 dimension is retained in
# the config and filename so this launcher follows the common naming scheme.

GPU_IDS="${1:-${CUDA_VISIBLE_DEVICES:-}}"

export lightx2v_path="${lightx2v_path:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export model_path="${model_path:-/data/nvme0/lhd_codes/HunyuanImage-3.0-instruct/HunyuanImage-3-Instruct}"
export HUNYUAN_IMAGE3_REPO_PATH="${HUNYUAN_IMAGE3_REPO_PATH:-/data/nvme0/lhd_codes/HunyuanImage-3.0}"
export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"

config_json="${CONFIG_JSON:-${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_ti2t_tp_sp_cfg_flashinfer.json}"

parallel_values=$(python - "$config_json" "$model_path" "ti2t" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
model_path = Path(sys.argv[2])
task = sys.argv[3]

with config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)

parallel = config.get("parallel") or {}
tp_size = int(parallel.get("tensor_p_size", 1))
sp_size = int(parallel.get("seq_p_size", 1))
cfg_size = int(parallel.get("cfg_p_size", 1))
attn_type = str(parallel.get("seq_p_attn_type", "kv_all_gather")).strip().lower().replace("-", "_")
cfg_mode = str(parallel.get("cfg_mode", config.get("hunyuan_cfg_mode", "batch"))).strip().lower()
pipeline_parallel = parallel.get("pipeline_parallel", config.get("pipeline_parallel", True))
moe_impl = str(config.get("moe_impl", "eager")).strip().lower()

if attn_type in ("kv_allgather", "kv_gather"):
    attn_type = "kv_all_gather"
if tp_size <= 1 or sp_size <= 1:
    raise SystemExit(f"{task} TP+SP launcher requires tensor_p_size > 1 and seq_p_size > 1, got TP={tp_size}, SP={sp_size}.")
if pipeline_parallel:
    raise SystemExit(f"{task} TP+SP launcher requires parallel.pipeline_parallel=false.")
if moe_impl != "flashinfer":
    raise SystemExit(f"{task} FlashInfer launcher requires moe_impl='flashinfer', got {moe_impl!r}.")
if attn_type not in ("kv_all_gather", "ulysses"):
    raise SystemExit(f"Unsupported seq_p_attn_type: {attn_type}.")
if config.get("enable_cfg", False) or cfg_size != 1:
    raise SystemExit(
        f"{task} does not support diffusion CFG and requires enable_cfg=false and cfg_p_size=1; "
        f"got enable_cfg={config.get('enable_cfg', False)}, cfg_p_size={cfg_size}."
    )
if cfg_mode not in ("batch", "serial"):
    raise SystemExit(f"{task} requires a non-parallel CFG mode, got cfg_mode={cfg_mode!r}.")

if attn_type == "ulysses":
    model_config_path = model_path / "config.json"
    if not model_config_path.is_file():
        raise SystemExit(f"Cannot validate Ulysses heads because model config does not exist: {model_config_path}.")
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)
    q_heads = int(model_config.get("num_attention_heads") or model_config["num_heads"])
    kv_heads = int(model_config.get("num_key_value_heads") or q_heads)
    combined_size = tp_size * sp_size
    if q_heads % combined_size or kv_heads % combined_size:
        raise SystemExit(
            "Ulysses requires TP_SIZE * SP_SIZE to divide Q and KV heads: "
            f"Q={q_heads}, KV={kv_heads}, TP={tp_size}, SP={sp_size}."
        )

print(tp_size, sp_size, cfg_size, attn_type)
PY
)
read -r TP_SIZE SP_SIZE CFG_SIZE SP_ATTN_TYPE <<< "$parallel_values"
WORLD_SIZE=$((TP_SIZE * SP_SIZE * CFG_SIZE))

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -ge "$WORLD_SIZE" ]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((WORLD_SIZE - 1)))
    fi
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "No CUDA devices were selected; ${WORLD_SIZE} visible GPUs are required." >&2
    exit 2
fi

IFS=',' read -r -a raw_visible_gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
visible_gpu_ids=()
declare -A seen_gpu_ids=()
for gpu_id in "${raw_visible_gpu_ids[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [ -z "$gpu_id" ]; then
        echo "CUDA_VISIBLE_DEVICES contains an empty GPU id." >&2
        exit 2
    fi
    if [ -n "${seen_gpu_ids[$gpu_id]:-}" ]; then
        echo "Duplicate GPU id in CUDA_VISIBLE_DEVICES: $gpu_id" >&2
        exit 2
    fi
    seen_gpu_ids[$gpu_id]=1
    visible_gpu_ids+=("$gpu_id")
done

if [ "${#visible_gpu_ids[@]}" -ne "$WORLD_SIZE" ]; then
    echo "HunyuanImage3 ${TP_SIZE}x${SP_SIZE}x${CFG_SIZE} parallelism requires exactly ${WORLD_SIZE} visible GPUs; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${visible_gpu_ids[*]}")

image_path="${image_path:-${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_0_0.png}"
prompt="${prompt:-请描述图像中的主要内容和视觉风格。}"
seed="${seed:-42}"
save_result_path="${SAVE_RESULT_PATH:-${save_result_path:-${lightx2v_path}/save_results/hunyuan_image3_ti2t_tp2_sp2_cfg1_flashinfer.txt}}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Running HunyuanImage3 ti2t with TP_SIZE=${TP_SIZE}, SP_SIZE=${SP_SIZE}, CFG_SIZE=${CFG_SIZE}, SP_ATTN_TYPE=${SP_ATTN_TYPE}"
echo "config_json=${config_json}"

source "${lightx2v_path}/scripts/base/base.sh"

torchrun \
    --standalone \
    --nproc_per_node="$WORLD_SIZE" \
    --max_restarts=0 \
    -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task ti2t \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "$prompt" \
    --image_path "$image_path" \
    --save_result_path "$save_result_path" \
    --seed "$seed"
