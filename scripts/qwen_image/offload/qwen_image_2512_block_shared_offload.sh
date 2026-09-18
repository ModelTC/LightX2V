#!/bin/bash
set -eo pipefail

# Resolve user paths before switching to the repository root.
caller_dir="${PWD}"
absolute_path() {
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "${caller_dir}" "$1" ;; esac
}

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="$(absolute_path "${QWEN_MODEL_PATH:-${lightx2v_path}/models/Qwen-Image-2512}")"
task="${TASK:-t2i}"
if [[ "${task}" != t2i ]]; then
    echo "The Qwen BF16 shared-weight adapter currently supports TASK=t2i only." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
nproc="${#gpu_ids[@]}"
case "${nproc}" in
    1|2|3|4|6|8|12|24) ;;
    *) echo "Unsupported GPU count ${nproc}; supported: 1, 2, 3, 4, 6, 8, 12, 24." >&2; exit 2 ;;
esac
config_template="$(absolute_path "${CONFIG_JSON:-${lightx2v_path}/configs/qwen_image/offload/qwen_image_2512_block_shared.json}")"

config_json="$(mktemp "${TMPDIR:-/tmp}/lightx2v-qwen_image-XXXXXX.json")"
trap 'rm -f -- "${config_json}"' EXIT

python - "${config_template}" "${config_json}" "${nproc}" <<'PY_CHECK'
import json
import math
import os
import sys

path, output, count = sys.argv[1:]
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
    config["parallel"]["seq_p_size"] = count
size = math.prod(config["parallel"].get(key, 1) for key in ("tensor_p_size", "seq_p_size", "cfg_p_size"))
if size != count or size != len(devices):
    raise SystemExit(f"Config TP*SP*CFG={size} must equal the number of visible GPUs ({len(devices)})")
with open(output, "w") as destination:
    json.dump(config, destination, indent=2)
    destination.write("\n")
print(f"GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}; scope: {scope}; config: {output}")
PY_CHECK

export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16
source "${lightx2v_path}/scripts/base/base.sh"

default_prompt='A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.'
cd -- "${lightx2v_path}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${nproc}" -m lightx2v.infer \
  --model_cls qwen_image \
  --task "${task}" \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${QWEN_PROMPT:-${default_prompt}}" \
  --negative_prompt " " \
  --save_result_path "$(absolute_path "${QWEN_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/qwen_image_t2i_2512_block_shared_offload.png}")" \
  --seed "${SEED:-42}"
