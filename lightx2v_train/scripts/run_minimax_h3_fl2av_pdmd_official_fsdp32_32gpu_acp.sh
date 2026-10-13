#!/usr/bin/env bash
set -Eeuo pipefail

# Same command on four ACP workers, eight GPUs each. No training in --dry-run.
steps="${1:?Pass the rollout step count (4 or 8), optionally followed by --dry-run}"
shift
if [[ ! "$steps" =~ ^(4|8)$ || $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: $0 {4|8} [--dry-run]" >&2
    exit 1
fi
: "${MASTER_ADDR:?ACP must set the same MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set the same MASTER_PORT on all four workers}"
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
config="$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_fl2av_pdmd${steps}_official_fsdp32.yaml"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_FL2AV_CACHE="${H3_FL2AV_CACHE:-${H3_MULTITASK_3TASK_PORTRAIT_VIEW:-/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_dmd_cache_5s_768p/task_aware_view_3task_portrait}}"
export H3_FL2AV_PDMD_OUTPUT="${H3_FL2AV_PDMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_fl2av_pdmd${steps}_official_lorafp32_32gpu}"

if [[ -z "${LOCAL_KERNELS:-}" ]]; then
    if [[ -z "${H3_KERNEL_SNAPSHOT:-}" && -n "${KERNELS_CACHE:-}" ]]; then
        H3_KERNEL_SNAPSHOT="$KERNELS_CACHE/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b"
    fi
    if [[ -n "${H3_KERNEL_SNAPSHOT:-}" ]]; then
        [[ -d "$H3_KERNEL_SNAPSHOT" ]] || { echo "Missing FlashAttention-3 snapshot: $H3_KERNEL_SNAPSHOT" >&2; exit 1; }
        export LOCAL_KERNELS="kernels-community/flash-attn3=$H3_KERNEL_SNAPSHOT"
    fi
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HOME="${HF_HOME:-/tmp/h3_hf_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3_fl2av_pdmd_pycache}"

# Metadata-only preflight: legacy task views do not have Omni merge receipts.
"$H3_PYTHON" - "$config" "$steps" <<'PY'
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

config_module = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) / "lightx2v_train/runtime/config.py"
spec = importlib.util.spec_from_file_location("_fl2av_config", config_module)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.load_config(sys.argv[1])
model, training, data = config["model"], config["training"], config["data"]["train"]
if model["name"] != "minimax_h3_t2av" or data.get("tasks") != ["fl2av"]:
    raise SystemExit("This launcher requires the FL2AV-only base-transformer recipe.")
if not training["dmd"].get("official_pdmd") or training["dmd"]["num_inference_steps"] != int(sys.argv[2]):
    raise SystemExit("Official PDMD flow / rollout steps do not match this launcher.")
if not (Path(model["pretrained_model_name_or_path"]) / "transformer/config.json").is_file():
    raise SystemExit("H3_MODEL_PATH must contain transformer/config.json (FL2AV base, not transformer_ref).")
root = Path(data["data_path"])
manifests = [root / "fl2av/metadata.jsonl"]
for bucket in data["buckets"]:
    directory = root / "fl2av" / bucket
    manifests.append(directory / "metadata.jsonl")
    manifests.extend(sorted(directory.glob("*/metadata.jsonl")))
manifests = list(dict.fromkeys(path.resolve() for path in manifests if path.is_file()))
counts = Counter()
for manifest in manifests:
    first = True
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            height, width = int(row["target_height"]), int(row["target_width"])
            frames = int(row.get("target_num_frames", row.get("num_frames", 0)))
            if row.get("task") != "fl2av" or frames != 124 or [height, width] not in [[768, 1344], [1344, 768]]:
                raise SystemExit(f"Expected FL2AV 124-frame 768p cache: {manifest}")
            condition = Path(row["condition_path"])
            if not condition.is_absolute():
                condition = manifest.parent / condition
            if first and not condition.is_file():
                raise SystemExit(f"Missing cached condition: {condition}. Preserve/rebase cache paths on this machine.")
            first = False
            counts["landscape" if width > height else "portrait"] += 1
if any(counts[bucket] < 32 for bucket in data["buckets"]):
    raise SystemExit(f"Need at least 32 FL2AV rows per orientation: {dict(counts)}")
print(f"FL2AV cache: {dict(counts)}, total={sum(counts.values())}; T2AV/I2AV excluded")
print(f"Official PDMD: steps={sys.argv[2]}, 4 x 8 GPUs, global_batch=32, landscape=16, portrait=16")
print(f"optimizer_updates={training['max_train_iters']}, save_every_iters={training['save_every_iters']}, save_total_limit={training['save_total_limit']}")
print("5 critic updates -> 1 student update; student reuses the preceding critic trajectory, without reading a new batch.")
print("Precision: student base BF16 + LoRA FP32; teacher BF16; fake full FP32; FSDP BF16 compute / FP32 reduction.")
print(f"config={sys.argv[1]}\ndata={root}\noutput={training['output_dir']}")
print("Use a NEW output directory, not a Ref2AV, legacy DMD, or different-step PDMD checkpoint directory.")
PY

cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=4 --nproc_per_node=8
    "--rdzv_id=${H3_RDZV_ID:-h3_fl2av_pdmd${steps}_official_lorafp32_32gpu}"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$config"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
