#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/mnt/devsft_afs_2/gushiqiao/ar/models/lingbot-world-v2-14b-causal-fast}
QUANT_DEVICE=${QUANT_DEVICE:-cpu}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT_FILE="$MODEL_DIR/quantized/lingbot_fast_v2_fp8_f16_accum_qmax14.safetensors"

if [[ -f "$OUTPUT_FILE" ]]; then
    python - "$OUTPUT_FILE" <<'PY'
import sys
from safetensors import safe_open

with safe_open(sys.argv[1], framework="pt", device="cpu") as checkpoint:
    metadata = checkpoint.metadata()
    assert metadata["quantization_profile"] == "wan-ar-fp8-f16-accum", metadata
    assert metadata["model_type"] == "lingbot_fast", metadata
    assert float(metadata["weight_qmax"]) == 14.0, metadata
    assert int(metadata["quantized_linears"]) == 563, metadata
print("Validated existing reduced-range LingBot v2 checkpoint:", sys.argv[1])
PY
    exit 0
fi

cd "$REPO_ROOT"
python tools/convert/converter.py \
    --source "$MODEL_DIR/transformers" \
    --output "$MODEL_DIR/quantized" \
    --output_name lingbot_fast_v2_fp8_f16_accum_qmax14 \
    --model_type lingbot_fast --quantized --linear_type fp8 \
    --quantization_profile wan-ar-fp8-f16-accum \
    --single_file --non_linear_dtype torch.bfloat16 --device "$QUANT_DEVICE"
