#!/bin/bash
set -eo pipefail

# Resolve user paths before switching to the repository root.
caller_dir="${PWD}"
absolute_path() {
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "${caller_dir}" "$1" ;; esac
}

task="${TASK:-t2i}"
case "${task}" in t2i|ti2i) ;; *) echo "Unsupported TASK: ${task}; choose t2i or ti2i." >&2; exit 2 ;; esac
for argument in "$@"; do
    case "${argument%%=*}" in
        --config_json|--model_cls|--task)
            echo "Use CONFIG_JSON and TASK to select the launch configuration and task." >&2
            exit 2 ;;
    esac
done
lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="$(absolute_path "${HUNYUAN_IMAGE3_MODEL_PATH:-${lightx2v_path}/../HunyuanImage-3-Instruct}")"
export HUNYUAN_IMAGE3_REPO_PATH="$(absolute_path "${HUNYUAN_IMAGE3_REPO_PATH:-${lightx2v_path}/../HunyuanImage-3.0}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
nproc="${#gpu_ids[@]}"
case "${nproc}" in
    1|2|4|8|16) ;;
    *) echo "Unsupported GPU count ${nproc}; supported: 1, 2, 4, 8, 16." >&2; exit 2 ;;
esac
config_template="$(absolute_path "${CONFIG_JSON:-${lightx2v_path}/configs/hunyuan_image3/offload/hunyuan_image3_block_shared.json}")"

config_json="$(mktemp "${TMPDIR:-/tmp}/lightx2v-hunyuan_image3-XXXXXX.json")"
trap 'rm -f -- "${config_json}"' EXIT

python - "${config_template}" "${config_json}" "${nproc}" "${task}" <<'PY_CHECK'
import json
import math
import os
import sys

path, output, count, task = sys.argv[1:]
count = int(count)
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if any(not item or item.strip() != item or item == "-1" for item in devices) or len(set(devices)) != len(devices):
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain distinct, nonempty GPU IDs without spaces")
with open(path) as source:
    config = json.load(source)
if not (config["cpu_offload"] and config["shared_cpu_weights"] and config["offload_granularity"] == "block"):
    raise SystemExit("Expected CPU block offload with shared CPU weights")
scope = os.environ.get("SHARED_CPU_WEIGHT_SCOPE", config["shared_cpu_weight_scope"])
if scope not in ("host", "numa"):
    raise SystemExit("SHARED_CPU_WEIGHT_SCOPE must be host or numa")
config["shared_cpu_weight_scope"] = scope
# Explicit config overrides retain their configured topology.
if not os.environ.get("CONFIG_JSON"):
    tp, sp, cfg = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2), 16: (2, 4, 2)}[count]
    config["parallel"].update(tensor_p_size=tp, seq_p_size=sp, cfg_p_size=cfg, cfg_mode="parallel" if cfg == 2 else "serial")
    config["flashinfer_autotune_cache"] = f"save_results/hunyuan_image3_flashinfer_autotune_{task}_tp{tp}_sp{sp}_cfg{cfg}.json"
    if task == "ti2i":
        config.pop("size", None)
        config.update(image_size="auto", align_image_size=True)
size = math.prod(config["parallel"].get(key, 1) for key in ("tensor_p_size", "seq_p_size", "cfg_p_size"))
if size != count or size != len(devices):
    raise SystemExit(f"Config TP*SP*CFG={size} must equal the number of visible GPUs ({len(devices)})")
with open(output, "w") as destination:
    json.dump(config, destination, indent=2)
    destination.write("\n")
print(f"GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}; scope: {scope}; config: {output}")
PY_CHECK

export PYTHONPATH="${HUNYUAN_IMAGE3_REPO_PATH}:${PYTHONPATH:-}"
export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None
source "${lightx2v_path}/scripts/base/base.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ "${task}" == ti2i ]]; then
    default_prompt='新年宠物海报，Q版圆润的可爱标题“新年快乐汪”，副标题“HAPPY NEW YEAR”。鱼眼镜头，背景是房间门口，上传的主体歪头笑，围着红色围巾，戴着红色毛线帽，高清绒毛细节，面部特写，宝丽莱相纸，写实胶片摄影，复古颗粒感。'
    image_args=(--image_path "$(absolute_path "${HUNYUAN_IMAGE3_IMAGE_PATH:-${HUNYUAN_IMAGE3_REPO_PATH}/assets/demo_instruct_imgs/input_0_0.png}")")
else
    default_prompt='生成图片：一辆汽车行驶在高速公路上，驾驶员在打电话，副驾驶坐着一只狗'
    image_args=()
fi
save_path="$(absolute_path "${HUNYUAN_IMAGE3_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/hunyuan_image3_${task}_block_shared_offload.png}")"
mkdir -p -- "$(dirname -- "${save_path}")"
cd -- "${lightx2v_path}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${nproc}" -m lightx2v.infer \
    --model_cls hunyuan_image3 \
    --task "${task}" \
    --model_path "${model_path}" \
    --config_json "${config_json}" \
    --prompt "${HUNYUAN_IMAGE3_PROMPT:-${default_prompt}}" \
    "${image_args[@]}" \
    --save_result_path "${save_path}" \
    --seed "${SEED:-42}" \
    "$@"
