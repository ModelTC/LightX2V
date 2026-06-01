#!/bin/bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lightx2v_path=${LIGHTX2V_PATH:-$(cd "${script_dir}/../.." && pwd)}
model_path=${MODEL_PATH:-${1:-}}

if [[ -z "${model_path}" ]]; then
    echo "Usage: MODEL_PATH=/path/to/Helios-Distilled $0"
    echo "   or: $0 /path/to/Helios-Distilled"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls helios_distilled \
--task t2v \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/helios/helios_distilled_t2v.json" \
--prompt "A cinematic close-up of a snow leopard walking across a windy ridge at sunrise, detailed fur moving naturally in the light." \
--negative_prompt "overexposed, blurry, low quality, jpeg artifacts, static frame, distorted anatomy, extra limbs" \
--save_result_path "${lightx2v_path}/save_results/output_helios_distilled_t2v.mp4"
