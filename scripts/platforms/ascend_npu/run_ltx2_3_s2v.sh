#!/usr/bin/env bash
set -Eeuo pipefail

# LTX-2.3 speech-to-video inference on one Ascend NPU.
# MODEL_PATH points to the LTX-2 component directory; the LTX-2.3 DiT
# checkpoint is selected by CONFIG_PATH.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../../.." && pwd)}"
model_path="${MODEL_PATH:-/data/wushuo1/models/LTX-2}"
config_path="${CONFIG_PATH:-${lightx2v_path}/configs/platforms/ascend_npu/ltx2_3_s2v.json}"
audio_path="${AUDIO_PATH:-${lightx2v_path}/assets/inputs/audio/seko_input.mp3}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/output_lightx2v_ltx2_3_s2v_ascend.mp4}"
seed="${SEED:-42}"

prompt="${PROMPT:-A person speaks clearly in a quiet room, natural lighting, cinematic medium shot.}"
negative_prompt="${NEGATIVE_PROMPT:-blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts.}"

export PLATFORM="ascend_npu"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}"

if [[ ! "${ASCEND_RT_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
    echo "Error: this single-NPU script requires one numeric ASCEND_RT_VISIBLE_DEVICES value." >&2
    exit 64
fi
if [[ ! -d "${model_path}" ]]; then
    echo "Error: MODEL_PATH does not exist: ${model_path}" >&2
    exit 66
fi
if [[ ! -f "${config_path}" ]]; then
    echo "Error: CONFIG_PATH does not exist: ${config_path}" >&2
    exit 66
fi
if [[ ! -f "${audio_path}" ]]; then
    echo "Error: AUDIO_PATH does not exist: ${audio_path}" >&2
    exit 66
fi

mkdir -p "$(dirname -- "${save_result_path}")"

# LightX2V's base environment reads these lower-case variables.
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --seed "${seed}" \
    --model_cls ltx2 \
    --task ltx2_s2v \
    --model_path "${model_path}" \
    --config_json "${config_path}" \
    --audio_path "${audio_path}" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --save_result_path "${save_result_path}"
