#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same script on all four ACP workers (eight GPUs per worker).
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_CONFIG_PATH="${H3_CONFIG_PATH:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_dmd_lora_match124_image_audio_1to5_uniform_fsdp32_32gpu_full_fake.yaml}"
: "${MASTER_ADDR:?Set the ACP rendezvous host}"
: "${MASTER_PORT:?Set the ACP rendezvous port}"
: "${H3_MODEL_PATH:?Set the converted MiniMax-H3 model directory}"
: "${H3_REF2AV_CACHE:?Set the cached Ref2AV metadata.jsonl path}"
: "${H3_REF2AV_DMD_OUTPUT:?Set a shared output directory}"
export H3_MODEL_PATH H3_REF2AV_CACHE H3_REF2AV_DMD_OUTPUT
export H3_PDMD="${H3_PDMD:-false}"
case "$H3_PDMD" in
    true|false) ;;
    *) echo "H3_PDMD must be true or false." >&2; exit 1 ;;
esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi

if [[ ! -f "$H3_MODEL_PATH/transformer_ref/config.json" ]]; then
    echo "Missing Ref2AV transformer config: $H3_MODEL_PATH/transformer_ref/config.json" >&2
    exit 1
fi
if [[ ! -f "$H3_CONFIG_PATH" || ! -f "$H3_REF2AV_CACHE" ]]; then
    echo "The training config and Ref2AV cache manifest must exist." >&2
    exit 1
fi
H3_CONFIG_PATH="$(cd "$(dirname "$H3_CONFIG_PATH")" && pwd)/$(basename "$H3_CONFIG_PATH")"
command -v "$H3_PYTHON" >/dev/null
if [[ -n "${H3_REF2AV_EXPECTED_ROWS:-}" ]]; then
    actual_rows=$(wc -l < "$H3_REF2AV_CACHE")
    if [[ "$actual_rows" -ne "$H3_REF2AV_EXPECTED_ROWS" ]]; then
        echo "Cache rows: expected $H3_REF2AV_EXPECTED_ROWS, found $actual_rows." >&2
        exit 1
    fi
fi

# Reuse an offline FlashAttention-3 snapshot when supplied by the worker image.
if [[ -z "${H3_KERNEL_SNAPSHOT:-}" && -n "${KERNELS_CACHE:-}" ]]; then
    H3_KERNEL_SNAPSHOT="$KERNELS_CACHE/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b"
fi
if [[ -n "${H3_KERNEL_SNAPSHOT:-}" ]]; then
    [[ -d "$H3_KERNEL_SNAPSHOT" ]] || { echo "Missing FlashAttention-3 snapshot: $H3_KERNEL_SNAPSHOT" >&2; exit 1; }
    export LOCAL_KERNELS="kernels-community/flash-attn3=$H3_KERNEL_SNAPSHOT"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HOME="${HF_HOME:-/tmp/h3_hf_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3_pycache}"

# Resolve the actual config, including overrides, without importing torch/CUDA.
"$H3_PYTHON" - "$H3_CONFIG_PATH" <<'PY'
import os
import sys
from lightx2v_train.runtime.config import load_config

config = load_config(sys.argv[1])
projected = config["model"].get("capabilities", {}).get("distribution_matching", {}).get("projected_dmd", False)
expected = os.environ["H3_PDMD"] == "true"
dmd = config["training"]["dmd"]
order = dmd.get("update_order", "student_first")
ratio = int(dmd["fake_update_ratio"])
if projected is not expected:
    raise SystemExit("H3_PDMD disagrees with the selected config's projected_dmd.")
if expected and (order != "student_first" or ratio != 1):
    raise SystemExit("H3 PDMD requires update_order=student_first and fake_update_ratio=1.")
print(f"H3 Ref2AV: 4 x 8 GPUs, PDMD={str(projected).lower()}, update_order={order}, fake_update_ratio={ratio}, output={config['training']['output_dir']}")
PY

cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=4 --nproc_per_node=8
    "--rdzv_id=${H3_RDZV_ID:-h3_ref2av_fsdp32}"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$H3_CONFIG_PATH"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
