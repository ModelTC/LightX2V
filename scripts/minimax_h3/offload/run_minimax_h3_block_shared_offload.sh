#!/bin/bash
set -eo pipefail

# Resolve user paths before switching to the repository root.
caller_dir="${PWD}"
absolute_path() {
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "${caller_dir}" "$1" ;; esac
}

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="$(absolute_path "${MINIMAX_H3_MODEL_PATH:-${lightx2v_path}/models/MiniMax-H3}")"
task="${TASK:-t2av}"
case "${task}" in
    t2av)
        input_kinds=()
        default_prompt='integrated_multimodal_description: A cinematic fox walks through a snowy pine forest at dawn. overall_soundscape: Soft wind, crunching snow, and distant birds. non_diegetic_music: Quiet warm strings.' ;;
    i2av)
        input_kinds=(image)
        image_path="${MINIMAX_H3_IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/img_0.jpg}"
        default_prompt='Animate this image with natural synchronized sound.' ;;
    l2av)
        input_kinds=(last_frame)
        last_frame_path="${MINIMAX_H3_LAST_FRAME_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png}"
        default_prompt='Generate the preceding scene with natural synchronized sound.' ;;
    fl2av)
        input_kinds=(image last_frame)
        image_path="${MINIMAX_H3_IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png}"
        last_frame_path="${MINIMAX_H3_LAST_FRAME_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png}"
        default_prompt='Create a coherent transition with natural synchronized sound.' ;;
    ref2av)
        input_kinds=(image video audio)
        image_path="${MINIMAX_H3_IMAGE_PATH:-}"
        video_path="${MINIMAX_H3_VIDEO_PATH:-}"
        audio_path="${MINIMAX_H3_AUDIO_PATH:-}"
        if [[ -z "${image_path}" && -z "${video_path}" && -z "${audio_path}" ]]; then
            image_path="${lightx2v_path}/assets/inputs/imgs/img_0.jpg"
        fi
        if [[ -z "${image_path}" && -z "${video_path}" ]]; then
            echo "Ref2AV requires an image or video reference; audio-only input is unsupported." >&2
            exit 2
        fi
        default_prompt='Generate an audio-video scene following the references.' ;;
    *) echo "Unsupported TASK: ${task}; choose t2av, i2av, l2av, fl2av or ref2av." >&2; exit 2 ;;
esac
input_args=()
for kind in "${input_kinds[@]}"; do
    name="${kind}_path"
    [[ -n "${!name}" ]] || continue
    IFS=',' read -r -a paths <<< "${!name}"
    for index in "${!paths[@]}"; do
        paths[index]="$(absolute_path "${paths[index]}")"
    done
    input_args+=("--${kind}_path" "$(IFS=','; echo "${paths[*]}")")
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
nproc="${#gpu_ids[@]}"
case "${nproc}" in
    1|2|4|7|8|14|28|56) ;;
    *) echo "Unsupported GPU count ${nproc}; supported: 1, 2, 4, 7, 8, 14, 28, 56." >&2; exit 2 ;;
esac
config_template="$(absolute_path "${MINIMAX_H3_CONFIG:-${lightx2v_path}/configs/minimax_h3/offload/minimax_h3_block_shared_offload.json}")"

config_json="$(mktemp "${TMPDIR:-/tmp}/lightx2v-minimax_h3-XXXXXX.json")"
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
print(f"GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}; scope: {scope}; config: {output}")
PY_CHECK

export DTYPE=BF16 SENSITIVE_LAYER_DTYPE=BF16
source "${lightx2v_path}/scripts/base/base.sh"

# Ref2AV needs its own AdaLN cache; the other tasks share the FL2AV cache.
cd -- "${lightx2v_path}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${nproc}" -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task "${task}" \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${MINIMAX_H3_PROMPT:-${default_prompt}}" \
  "${input_args[@]}" \
  --save_result_path "$(absolute_path "${MINIMAX_H3_SAVE_RESULT_PATH:-${lightx2v_path}/save_results/minimax_h3_${task}_block_shared_offload.mp4}")" \
  --seed "${SEED:-42}"
