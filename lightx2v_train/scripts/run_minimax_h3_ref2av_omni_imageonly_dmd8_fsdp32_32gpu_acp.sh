#!/usr/bin/env bash
set -Eeuo pipefail

# ACP: run this identical command on each of four workers, eight GPUs each.
# MASTER_ADDR / MASTER_PORT must refer to the same reachable rendezvous host.
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_CONFIG_PATH="${H3_CONFIG_PATH:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_match124_bf16_fsdp32.yaml}"
: "${MASTER_ADDR:?ACP must set a common MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set a common MASTER_PORT on all four workers}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
export H3_REF2AV_DMD_OUTPUT="${H3_REF2AV_DMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_ref2av_omni_imageonly_dmd8_10k}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi
if [[ "${H3_PDMD:-false}" != false ]]; then
    echo "This is the standard DMD recipe; unset H3_PDMD or set it to false." >&2
    exit 1
fi
if [[ ! -f "$H3_MODEL_PATH/transformer_ref/config.json" ]]; then
    echo "Missing converted Ref2AV transformer config: $H3_MODEL_PATH/transformer_ref/config.json" >&2
    exit 1
fi
if [[ ! -f "$H3_CONFIG_PATH" || ! -s "$H3_REF2AV_CACHE" ]]; then
    echo "Training config and nonempty merged Ref2AV cache manifest must exist. Finish all four cache shards and merge first." >&2
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

# Resolve the actual YAML, including H3_CONFIG_PATH overrides, before torchrun.
"$H3_PYTHON" - "$H3_CONFIG_PATH" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from lightx2v_train.runtime.config import load_config

config = load_config(sys.argv[1])
training = config["training"]
dmd = training["dmd"]
distributed = config["distributed"]
data = config["data"]["train"]
sampler = data["reference_cost_sampler"]
matching = config["model"]["capabilities"]["distribution_matching"]
checks = {
    "standard DMD (not PDMD)": training["method"] == "dmd" and matching.get("projected_dmd", False) is False,
    "8 steps, 10000 iterations, grad accumulation 1": dmd["num_inference_steps"] == 8 and training["max_train_iters"] == 10000 and training["gradient_accumulation_iters"] == 1,
    "student-first, 5 critic updates": dmd["update_order"] == "student_first" and dmd["fake_update_ratio"] == 5,
    "FSDP32, no sequence parallel": distributed["fsdp2"]["enabled"] and distributed["fsdp2"]["size"] == 32 and not distributed["sequence_parallel"]["enabled"],
    "per-rank microbatch 1, image-count coverage": data["batch_size"] == 1 and sampler.get("batch_mode") == "count_coverage" and sampler.get("require_image_only") is True,
    "no category/orientation undersampling": sampler.get("balance_image_counts") is False and sampler.get("balance_orientation") is False,
    "metadata-controlled 124-frame generation": matching.get("geometry_from_metadata") is True and dmd["generation_shapes"] == [{"value": [124, 768, 1344]}],
}
for description, passed in checks.items():
    if not passed:
        raise SystemExit(f"Omni image-only recipe requires {description}.")
manifest = Path(data["data_path"])
if manifest.resolve() != Path(os.environ["H3_REF2AV_CACHE"]).resolve():
    raise SystemExit("Training config data_path disagrees with H3_REF2AV_CACHE.")
receipt_path = manifest.with_suffix(".complete.json")
if not receipt_path.is_file():
    raise SystemExit(f"Missing completed 32-shard merge receipt: {receipt_path}. Run merge_omni_ref2av_cache.py first.")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
counts = [receipt.get(name) for name in ("completed_count", "failed_count", "input_total_rows")]
if any(type(value) is not int or value < 0 for value in counts) or counts[0] == 0 or counts[0] + counts[1] != counts[2]:
    raise SystemExit(f"Invalid completed/failed/source accounting in {receipt_path}.")
if not receipt.get("preprocess_fingerprint"):
    raise SystemExit(f"Missing preprocessing fingerprint in {receipt_path}.")
digest = hashlib.sha256()
rows = 0
with manifest.open("rb") as handle:
    for line in handle:
        digest.update(line)
        rows += bool(line.strip())
if digest.hexdigest() != receipt.get("manifest_sha256") or rows != counts[0]:
    raise SystemExit("Cache manifest no longer matches its complete receipt; re-run the checked shard merge.")
print(f"Verified cache: completed={counts[0]}, failed={counts[1]}, input_rows={counts[2]}, ref_image_counts={receipt.get('reference_image_counts', {})}")
print(f"H3 Omni Ref2AV: 4 x 8 GPUs, standard DMD, steps=8, iters=10000, grad_accum=1, student_first, fake_update_ratio=5, global_microbatch=32, batch_mode=count_coverage, output={training['output_dir']}")
PY

cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=4 --nproc_per_node=8
    "--rdzv_id=${H3_RDZV_ID:-h3_ref2av_omni_imageonly_dmd8_10k}"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$H3_CONFIG_PATH"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
