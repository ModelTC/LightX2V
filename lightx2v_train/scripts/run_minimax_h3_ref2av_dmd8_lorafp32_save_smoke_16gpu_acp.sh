#!/usr/bin/env bash
set -Eeuo pipefail

# Run once on each of TWO ACP workers, eight GPUs each. No training restart:
# iter1 -> save1 -> iter2 -> save2 -> exit. Keep normal 32/64-GPU recipes intact.
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
export H3_CONFIG_PATH="${H3_DMD_SAVE_SMOKE_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_lorafp32_save_smoke_fsdp16.yaml}"
export H3_PDMD=false
unset H3_REF2AV_EXPECTED_ROWS
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
export H3_REF2AV_DMD_OUTPUT="${H3_REF2AV_DMD_OUTPUT:-$H3_CODE_ROOT/outputs/h3_dmd8_lorafp32_save_smoke_16gpu}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_dmd8_lorafp32_save_smoke_16gpu}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi
: "${MASTER_ADDR:?ACP must set the same MASTER_ADDR on both workers}"
: "${MASTER_PORT:?ACP must set the same MASTER_PORT on both workers}"
[[ "${NNODES:-2}" == 2 && "${NPROC_PER_NODE:-8}" == 8 ]] || { echo "This test needs NNODES=2 and NPROC_PER_NODE=8." >&2; exit 1; }
command -v "$H3_PYTHON" >/dev/null

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

# Read-only CPU preflight; do not load tensors or initialize CUDA.
"$H3_PYTHON" - "$H3_CONFIG_PATH" <<'PY'
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

module_path = Path(os.environ["H3_CODE_ROOT"]) / "lightx2v_train/lightx2v_train/runtime/config.py"
spec = importlib.util.spec_from_file_location("_h3_save_smoke_config", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.load_config(sys.argv[1])
training, model, distributed = config["training"], config["model"], config["distributed"]
dmd, data = training["dmd"], config["data"]["train"]
matching = model["capabilities"]["distribution_matching"]
checks = {
    "plain DMD8": training["method"] == "dmd" and dmd["num_inference_steps"] == 8 and not matching.get("projected_dmd", False) and not matching.get("official_pdmd", False) and not dmd.get("official_pdmd", False) and not dmd.get("residual_head", {}).get("enabled", False),
    "FSDP16 with SP disabled": distributed["fsdp2"]["enabled"] and distributed["fsdp2"]["size"] == 16 and not distributed["sequence_parallel"]["enabled"] and distributed["sequence_parallel"]["size"] == 1 and not distributed.get("dp", {}).get("enabled", False),
    "two iterations, save after each, no automatic resume": training["max_train_iters"] == 2 and training["save_every_iters"] == 1 and config.get("resume", {}).get("auto_resume") is False and not config.get("resume", {}).get("resume_path"),
    "per-rank batch1, accumulation1": data["batch_size"] == 1 and training["gradient_accumulation_iters"] == 1,
}
for description, passed in checks.items():
    if not passed:
        raise SystemExit(f"Save smoke test requires {description}.")
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if len(devices) != 8 or len(set(devices)) != 8 or any(not item.strip() for item in devices):
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain eight distinct GPUs per worker.")
for path, name in ((model["pretrained_model_name_or_path"], "H3_MODEL_PATH"), (data["data_path"], "H3_REF2AV_CACHE"), (training["output_dir"], "H3_REF2AV_DMD_OUTPUT")):
    if Path(path).resolve() != Path(os.environ[name]).resolve():
        raise SystemExit(f"Config disagrees with {name}.")
if not (Path(model["pretrained_model_name_or_path"]) / "transformer_ref/config.json").is_file():
    raise SystemExit("Missing Ref2AV model transformer_ref/config.json.")
output = Path(training["output_dir"])
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit("Use a NEW empty output directory for this save test; existing files are not overwritten or resumed.")
manifest = Path(data["data_path"])
receipt_path = manifest.with_suffix(".complete.json")
if not receipt_path.is_file():
    raise SystemExit(f"Missing completed cache merge receipt: {receipt_path}.")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
counts = [receipt.get(key) for key in ("completed_count", "failed_count", "input_total_rows")]
if any(type(value) is not int or value < 0 for value in counts) or counts[0] == 0 or sum(counts[:2]) != counts[2] or not receipt.get("preprocess_fingerprint"):
    raise SystemExit("Invalid completed cache receipt accounting/fingerprint.")
digest, rows = hashlib.sha256(), 0
with manifest.open("rb") as handle:
    for line in handle:
        digest.update(line)
        rows += bool(line.strip())
if digest.hexdigest() != receipt.get("manifest_sha256") or rows != counts[0]:
    raise SystemExit("Cache manifest no longer matches its complete receipt.")
print(f"Verified cache: completed={counts[0]}, failed={counts[1]}, input_rows={counts[2]}")
print("SAVE SMOKE: 2 x 8 GPUs, FSDP16, plain DMD8; iter1 -> save1 -> iter2 -> save2 -> exit")
print(f"config={sys.argv[1]}\ndata={manifest}\noutput={output}")
print(f"student LoRA param_dtype={training['student']['lora'].get('param_dtype', 'inherit_base')}; fake_update_ratio={dmd['fake_update_ratio']}; save_total_limit={training['save_total_limit']}")
print("Each outer iteration contains one student and five fake updates in the default recipe. This tests continuation after saving, NOT a restart/resume.")
print("FSDP16 holds larger local parameter/optimizer shards than FSDP32. A training OOM before save does not test checkpoint memory.")
PY

cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=2 --nproc_per_node=8
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
