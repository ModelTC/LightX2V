#!/bin/bash
set -eo pipefail

# Resolve user paths before switching to the repository root.
caller_dir="${PWD}"
absolute_path() {
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "${caller_dir}" "$1" ;; esac
}

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="$(absolute_path "${MINIMAX_H3_MODEL_PATH:-${lightx2v_path}/models/MiniMax-H3}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
nproc="${#gpu_ids[@]}"
case "${nproc}" in
    1|2|4|7|8|14|28|56) ;;
    *) echo "Unsupported GPU count ${nproc}; supported: 1, 2, 4, 7, 8, 14, 28, 56." >&2; exit 2 ;;
esac
config_template="$(absolute_path "${MINIMAX_H3_CONFIG:-${lightx2v_path}/configs/minimax_h3/offload/minimax_h3_t2av_block_shared_offload_numa_auto.json}")"

config_json="$(mktemp "${TMPDIR:-/tmp}/lightx2v-minimax_h3-XXXXXX.json")"
trap 'rm -f -- "${config_json}"' EXIT

python - "${config_template}" "${config_json}" "${nproc}" "numa" <<'PY_CHECK'
import json
import math
import os
import sys

path, output, count, scope = sys.argv[1:]
count = int(count)
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if any(not item or item.strip() != item or item == "-1" for item in devices) or len(set(devices)) != len(devices):
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain distinct, nonempty GPU IDs without spaces")
with open(path) as source:
    config = json.load(source)
if not (config["cpu_offload"] and config["shared_cpu_weights"] and config["offload_granularity"] == "block" and config["shared_cpu_weight_scope"] == scope):
    raise SystemExit(f"Expected CPU block offload with shared_cpu_weight_scope={scope!r}")
# Explicit config overrides retain their configured topology.
if not os.environ.get("MINIMAX_H3_CONFIG"):
    config["parallel"]["seq_p_size"] = count
    if count != 8:
        config["vae_decode_parallel"] = False
size = math.prod(config["parallel"].get(key, 1) for key in ("tensor_p_size", "seq_p_size", "cfg_p_size"))
if size != count or size != len(devices):
    raise SystemExit(f"Config TP*SP*CFG={size} must equal the number of visible GPUs ({len(devices)})")
with open(output, "w") as destination:
    json.dump(config, destination, indent=2)
    destination.write("\n")
PY_CHECK

echo "GPUs: ${CUDA_VISIBLE_DEVICES}; config: ${config_json}"

export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16
source "${lightx2v_path}/scripts/base/base.sh"

# Reuse the 29-step FL2AV AdaLN cache shared by the four base tasks.
cd -- "${lightx2v_path}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${nproc}" -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task l2av \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${MINIMAX_H3_PROMPT:-Generate the preceding scene with natural synchronized sound.}" \
  --last_frame_path "$(absolute_path "${MINIMAX_H3_LAST_FRAME_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png}")" \
  --save_result_path "$(absolute_path "${MINIMAX_H3_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/minimax_h3_l2av_block_shared_offload_numa.mp4}")" \
  --seed 42
