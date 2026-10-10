#!/usr/bin/env bash
set -Eeuo pipefail

# Run this identical script on all eight ACP workers (eight GPUs each).
# Independent 8 x 8 GPU entry point. Keep the 32-GPU launcher unchanged.
# Separate recipe/output: old shift12 DMD semantics, NEW Omni condition cache.
# Student LoRA FP32 is selected by training.student.lora.param_dtype in YAML;
# omit/set null to inherit the adapter dtype, or set bf16 to select BF16.
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
# Dedicated override prevents a stale H3_CONFIG_PATH from selecting PDMD/HEAD.
export H3_CONFIG_PATH="${H3_LEGACY_SHIFT12_FSDP64_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp64.yaml}"
export H3_PDMD=false
# The verified Omni merge receipt below owns row accounting. Do not reuse an
# exported EXPECTED_ROWS=13553 from the old image+audio launcher.
unset H3_REF2AV_EXPECTED_ROWS
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
export H3_REF2AV_DMD_OUTPUT="${H3_REF2AV_DMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp64_lora_fp32_100k}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp64_lora_fp32_100k}"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi
: "${MASTER_ADDR:?ACP must set a common MASTER_ADDR on all eight workers}"
: "${MASTER_PORT:?ACP must set a common MASTER_PORT on all eight workers}"

# c10d assigns node ranks when ACP does not supply one. Never silently treat a
# possibly global RANK as a node rank. Explicit node-rank aliases must agree.
node_rank=""
for name in NODE_RANK GROUP_RANK ACP_NODE_RANK; do
    value="${!name-}"
    if [[ -n "$value" ]]; then
        [[ "$value" =~ ^[0-7]$ ]] || { echo "$name must be a node rank in 0..7." >&2; exit 1; }
        [[ -z "$node_rank" || "$value" == "$node_rank" ]] || { echo "Conflicting ACP node ranks: $name=$value, previous=$node_rank." >&2; exit 1; }
        node_rank="$value"
    fi
done
for name in NNODES NPROC_PER_NODE; do
    value="${!name-}"
    [[ -z "$value" || "$value" == 8 ]] || { echo "$name must be 8 for this 64-GPU launcher." >&2; exit 1; }
done
[[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$MASTER_PORT > 0 && 10#$MASTER_PORT <= 65535 )) || { echo "MASTER_PORT must be an integer in 1..65535." >&2; exit 1; }

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

# Validate the NEW merged cache without loading latent tensors, constructing
# models or changing data. Keep the integrity check of the other Omni launchers.
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
model = config["model"]
matching = model["capabilities"]["distribution_matching"]
data = config["data"]["train"]
sampler = data["reference_cost_sampler"]
distributed = config["distributed"]
fsdp = distributed["fsdp2"]
sp = distributed.get("sequence_parallel", {})
if fsdp.get("enabled") is not True or fsdp.get("size") != 64 or sp.get("enabled", False) or sp.get("size", 1) != 1:
    raise SystemExit("This launcher requires FSDP64, DP64 and no sequence parallel (size=1).")
if distributed.get("dp", {}).get("enabled", False):
    raise SystemExit("This launcher requires FSDP64, not DP/DDP wrapping.")
if dmd["num_inference_steps"] != 8 or data.get("batch_size", 1) != 1:
    raise SystemExit("This launcher requires DMD8 and per-rank microbatch 1.")
visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if len(visible_devices) != 8 or len(set(visible_devices)) != 8 or any(not value.strip() for value in visible_devices):
    raise SystemExit("CUDA_VISIBLE_DEVICES must list eight distinct GPUs per ACP node.")
if training["method"] != "dmd" or matching.get("projected_dmd", False) or matching.get("official_pdmd", False) or dmd.get("official_pdmd", False) or dmd.get("residual_head", {}).get("enabled", False):
    raise SystemExit("This launcher runs plain DMD, not PDMD or HEAD; select the appropriate launcher for those algorithms.")
manifest = Path(data["data_path"])
if manifest.resolve() != Path(os.environ["H3_REF2AV_CACHE"]).resolve():
    raise SystemExit("Training config data_path disagrees with H3_REF2AV_CACHE.")
receipt_path = manifest.with_suffix(".complete.json")
if not receipt_path.is_file():
    raise SystemExit(f"Missing completed cache merge receipt: {receipt_path}. Run merge_omni_ref2av_cache.py first.")
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
print(
    f"H3 Omni legacy-shift12: 8 x 8 GPUs, FSDP64/DP64/SP1, plain DMD, steps={dmd['num_inference_steps']}, "
    f"iters={training['max_train_iters']}, grad_accum={training['gradient_accumulation_iters']}, "
    f"legacy_numerics={matching.get('legacy_numerics', False)}, "
    f"model_mode={dmd.get('model_mode', 'eval')}, update_order={dmd.get('update_order', 'student_first')}, "
    f"fake_update_ratio={dmd['fake_update_ratio']}, CFG={training['teacher']['guidance_scale']}"
)
print(f"config={sys.argv[1]}\ndata={manifest}\noutput={training['output_dir']}")
print("legacy_numerics aligns condition noise and scalar sigma arithmetic; raw x0 = xt + sigma * velocity, without explicit dtype casts.")
print(f"sampler={json.dumps(sampler, sort_keys=True)}; workers={data['num_workers']}, pin_memory={data['pin_memory']}")
print(
    "Default sampling: one of the observed 1..6 image counts per 64-row global microbatch; "
    "Random orientation with no landscape/portrait quota. "
    "Balance observed image counts over the data epoch, rotating the omitted majority rows; "
    "NOT a full-data pass and NOT mixed-count coverage per batch. "
    "Each observed image count needs at least 64 rows."
)
for name in ("student", "fake", "teacher"):
    effective = model if name == "student" else {**model, **model.get(name, {})}
    role_mixed = model.get(name, {}).get("distributed", {}).get("fsdp2", {}).get("mixed_precision", {}) if name != "student" else {}
    mixed = {**fsdp.get("mixed_precision", {}), **role_mixed}
    print(f"precision {name}: transformer_param_dtype={effective['transformer_param_dtype']}, running_dtype={effective['running_dtype']}, fsdp={json.dumps(mixed, sort_keys=True)}")
print(f"student LoRA param_dtype={training['student'].get('lora', {}).get('param_dtype')} (null/omitted inherits the adapter dtype)")
for name in ("student", "fake"):
    print(f"{name}: {json.dumps(training[name], sort_keys=True)}")
print("Use a NEW output directory; old-trainer/PDMD/HEAD checkpoints are not a fresh aligned DMD run.")
PY

# This is deliberately standalone: the shared launcher fixes its topology at 32.
cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=8 --nproc_per_node=8
)
if [[ -n "$node_rank" ]]; then
    command+=("--node_rank=$node_rank")
fi
command+=(
    "--rdzv_id=$H3_RDZV_ID"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$H3_CONFIG_PATH"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
