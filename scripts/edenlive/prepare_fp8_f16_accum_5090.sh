#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model_root="${EDEN_MODEL:-${EDEN_X2V_MODEL_ROOT:-${repo_root}/models/EdenLive-x2v}}"
python_bin="${PYTHON_BIN:-python}"
quant_device="${QUANT_DEVICE:-cuda:0}"
proxy_quant_scheme="${PROXY_QUANT_SCHEME:-int8-convrot}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"
mkdir -p "${model_root}/quantized"

# Existing files must satisfy this profile; ordinary full-range FP8 files are
# incompatible with FP16 accumulation and must never be silently reused.
prepare_checkpoint() {
    local checkpoint_path="$1" profile="$2"
    shift 2
    if [[ -f "${checkpoint_path}" ]]; then
        "${python_bin}" - "${checkpoint_path}" "${profile}" <<'PY'
import sys
from safetensors import safe_open

with safe_open(sys.argv[1], framework="pt", device="cpu") as checkpoint:
    metadata = checkpoint.metadata() or {}
if sys.argv[2] == "edenlive-proxy-int8-convrot":
    if metadata.get("quant_scheme") != "int8-convrot" or metadata.get("group_size") != "256":
        raise ValueError(f"Incompatible existing INT8 ConvRot Proxy checkpoint: {sys.argv[1]}")
    print(f"Validated existing checkpoint: {sys.argv[1]}", flush=True)
    sys.exit(0)
if metadata.get("quantization_profile") != sys.argv[2] or float(metadata.get("weight_qmax", "nan")) != 14.0:
    raise ValueError(f"Incompatible existing checkpoint: {sys.argv[1]}; expected {sys.argv[2]} with weight_qmax=14")
if sys.argv[2] != "wan-ar-fp8-f16-accum" and float(metadata.get("activation_qmax", "nan")) != 7.0:
    raise ValueError(f"Incompatible existing activation range: {sys.argv[1]}; expected activation_qmax=7")
print(f"Validated existing checkpoint: {sys.argv[1]}", flush=True)
PY
    else
        "$@"
    fi
}

prepare_checkpoint "${model_root}/quantized/dit-5b-fp8-f16-accum-qmax14.safetensors" wan-ar-fp8-f16-accum \
    "${python_bin}" tools/convert/convert_edenlive.py \
    --source "${model_root}/dit-5b/diffusion_pytorch_model.safetensors" \
    --output "${model_root}/quantized" --output_name dit-5b-fp8-f16-accum-qmax14 \
    --quantize fp8-f16-accum --device "${quant_device}"

prepare_checkpoint "${model_root}/quantized/t5-fp8-f16-accum-qmax14.safetensors" wan-t5-fp8-f16-accum \
    "${python_bin}" tools/convert/converter.py \
    --source "${model_root}/umt5/text_encoder/model.safetensors" \
    --output "${model_root}/quantized" --output_name t5-fp8-f16-accum-qmax14 \
    --model_type wan_t5 --quantized --linear_type fp8 \
    --quantization_profile wan-t5-fp8-f16-accum \
    --non_linear_dtype torch.bfloat16 --device "${quant_device}" --single_file

case "${proxy_quant_scheme}" in
    int8-convrot)
        prepare_checkpoint "${model_root}/quantized/proxy-dino-int8-convrot.safetensors" edenlive-proxy-int8-convrot \
            "${python_bin}" tools/convert/convert_edenlive_proxy.py \
            --source "${model_root}/vision-encoder/projection.safetensors" \
            --output "${model_root}/quantized/proxy-dino-int8-convrot.safetensors" \
            --quant-scheme int8-convrot --device "${quant_device}" --group-size 256
        ;;
    fp8-f16-accum)
        prepare_checkpoint "${model_root}/quantized/proxy-dino-fp8-f16-accum-qmax14.safetensors" edenlive-proxy-fp8-f16-accum \
            "${python_bin}" tools/convert/convert_edenlive_proxy.py \
            --source "${model_root}/vision-encoder/projection.safetensors" \
            --output "${model_root}/quantized/proxy-dino-fp8-f16-accum-qmax14.safetensors" \
            --quant-scheme fp8-f16-accum --device "${quant_device}"
        ;;
    *)
        printf 'Unsupported PROXY_QUANT_SCHEME: %s\n' "${proxy_quant_scheme}" >&2
        exit 2
        ;;
esac
