#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/hunyuan_image3/run_hunyuan_image3_ti2i_tp_sp.sh
#   bash scripts/hunyuan_image3/run_hunyuan_image3_ti2i_tp_sp.sh ti2iv2
#   bash scripts/hunyuan_image3/run_hunyuan_image3_ti2i_tp_sp.sh 0,1,2,3,4,5,6,7 ti2iv2

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
DEMO="${DEMO:-ti2i}"

if [ $# -gt 0 ]; then
    case "$1" in
        ti2i|ti2iv2)
            DEMO="$1"
            ;;
        *)
            GPU_IDS="$1"
            ;;
    esac
fi

if [ $# -gt 1 ]; then
    DEMO="$2"
fi

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

if [ "$DEMO" = "ti2iv2" ]; then
    default_prompt="让图1的猫咪与图2的猫咪自拍，图1的猫咪说:“妈妈，我在乡下遇到了好朋喵”，背景为图3。"
    default_image_path="${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_2_0.png,${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_2_1.png,${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_2_2.png"
    default_seed=43
    default_save_result_path="${lightx2v_path}/save_results/hunyuan_image3_ti2iv3_tp_sp.png"
else
    default_prompt="新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。 鱼眼镜头，背景是房间门口，近景，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清，绒毛细节，面部特写。 宝丽莱相纸，超现实主义，写实主义，胶片摄影，打印颗粒感肌理。肌理，超写实，复古感。"
    default_image_path="${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_0_0.png"
    default_seed=42
    default_save_result_path="${lightx2v_path}/save_results/hunyuan_image3_ti2i_tp_sp.png"
fi

prompt="${prompt:-$default_prompt}"
image_path="${image_path:-$default_image_path}"
seed="${seed:-$default_seed}"
save_result_path="${SAVE_RESULT_PATH:-$default_save_result_path}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/hunyuan_image3/hunyuan_image3_ti2i_tp_sp.json}"

parallel_values=$(python - "$config_json" "$model_path" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
model_path = Path(sys.argv[2])
with config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)

parallel = config.get("parallel") or {}
tp_size = int(parallel.get("tensor_p_size", 1))
sp_size = int(parallel.get("seq_p_size", 1))
cfg_size = int(parallel.get("cfg_p_size", 1))
attn_type = str(parallel.get("seq_p_attn_type", "kv_all_gather")).strip().lower().replace("-", "_")
cfg_mode = str(parallel.get("cfg_mode", config.get("hunyuan_cfg_mode", "batch"))).strip().lower()
pipeline_parallel = parallel.get("pipeline_parallel", config.get("pipeline_parallel", True))
if attn_type in ("kv_allgather", "kv_gather"):
    attn_type = "kv_all_gather"

if tp_size <= 1 or sp_size <= 1:
    raise SystemExit(f"TP+SP launcher requires tensor_p_size > 1 and seq_p_size > 1, got TP={tp_size}, SP={sp_size}.")
if cfg_size not in (1, 2):
    raise SystemExit(f"TP+SP launcher requires cfg_p_size to be 1 or 2, got {cfg_size}.")
if pipeline_parallel:
    raise SystemExit("TP+SP launcher requires parallel.pipeline_parallel=false.")
if cfg_size == 2:
    if not config.get("enable_cfg", False):
        raise SystemExit("CFG parallel requires enable_cfg=true.")
    if cfg_mode != "parallel":
        raise SystemExit("CFG parallel requires parallel.cfg_mode='parallel'.")
elif config.get("enable_cfg", False) and cfg_mode != "serial":
    raise SystemExit("TP+SP with cfg_p_size=1 requires parallel.cfg_mode='serial'.")
if attn_type not in ("kv_all_gather", "ulysses"):
    raise SystemExit(f"Unsupported seq_p_attn_type: {attn_type}.")

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

IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-}"
visible_gpu_count=${#visible_gpu_ids[@]}
if [ "$visible_gpu_count" -ne "$WORLD_SIZE" ] || [ -z "${visible_gpu_ids[0]:-}" ]; then
    echo "HunyuanImage3 TP+SP requires exactly ${WORLD_SIZE} visible GPUs; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}" >&2
    exit 2
fi

declare -A seen_gpu_ids=()
for gpu_id in "${visible_gpu_ids[@]}"; do
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
done

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Running HunyuanImage3 ${DEMO} with TP_SIZE=${TP_SIZE}, SP_SIZE=${SP_SIZE}, CFG_SIZE=${CFG_SIZE}, SP_ATTN_TYPE=${SP_ATTN_TYPE}"
for ((cfg_rank=0; cfg_rank<CFG_SIZE; cfg_rank++)); do
    for ((sp_rank=0; sp_rank<SP_SIZE; sp_rank++)); do
        tp_group=()
        for ((tp_rank=0; tp_rank<TP_SIZE; tp_rank++)); do
            global_rank=$((((cfg_rank * SP_SIZE) + sp_rank) * TP_SIZE + tp_rank))
            tp_group+=("${global_rank}:${visible_gpu_ids[$global_rank]}")
        done
        group_csv=$(IFS=,; echo "${tp_group[*]}")
        echo "CFG rank ${cfg_rank}, SP rank ${sp_rank} / TP group (global_rank:gpu): ${group_csv}"
    done
done
for ((cfg_rank=0; cfg_rank<CFG_SIZE; cfg_rank++)); do
    for ((tp_rank=0; tp_rank<TP_SIZE; tp_rank++)); do
        sp_group=()
        for ((sp_rank=0; sp_rank<SP_SIZE; sp_rank++)); do
            global_rank=$((((cfg_rank * SP_SIZE) + sp_rank) * TP_SIZE + tp_rank))
            sp_group+=("${global_rank}:${visible_gpu_ids[$global_rank]}")
        done
        group_csv=$(IFS=,; echo "${sp_group[*]}")
        echo "CFG rank ${cfg_rank}, TP rank ${tp_rank} / SP group (global_rank:gpu): ${group_csv}"
    done
done
if [ "$CFG_SIZE" -gt 1 ]; then
    for ((sp_rank=0; sp_rank<SP_SIZE; sp_rank++)); do
        for ((tp_rank=0; tp_rank<TP_SIZE; tp_rank++)); do
            cfg_group=()
            for ((cfg_rank=0; cfg_rank<CFG_SIZE; cfg_rank++)); do
                global_rank=$((((cfg_rank * SP_SIZE) + sp_rank) * TP_SIZE + tp_rank))
                cfg_group+=("${global_rank}:${visible_gpu_ids[$global_rank]}")
            done
            group_csv=$(IFS=,; echo "${cfg_group[*]}")
            echo "SP rank ${sp_rank}, TP rank ${tp_rank} / CFG group (global_rank:gpu): ${group_csv}"
        done
    done
fi
echo "config_json=${config_json}"

source "${lightx2v_path}/scripts/base/base.sh"

torchrun \
    --standalone \
    --nproc_per_node="$WORLD_SIZE" \
    --max_restarts=0 \
    -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task ti2i \
    --model_path "$model_path" \
    --config_json "$config_json" \
    --prompt "$prompt" \
    --image_path "$image_path" \
    --save_result_path "$save_result_path" \
    --seed "$seed"
