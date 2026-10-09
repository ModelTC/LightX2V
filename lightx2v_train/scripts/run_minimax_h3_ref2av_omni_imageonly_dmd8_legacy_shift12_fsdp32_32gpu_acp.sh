#!/usr/bin/env bash
set -Eeuo pipefail

# Run this identical script on all four ACP workers (eight GPUs each).
# Separate recipe/output: old shift12 DMD semantics, NEW Omni condition cache.
export H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export H3_PYTHON="${H3_PYTHON:-python}"
# Dedicated override prevents a stale H3_CONFIG_PATH from selecting PDMD/HEAD.
export H3_CONFIG_PATH="${H3_LEGACY_SHIFT12_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp32.yaml}"
export H3_PDMD=false
# The verified Omni merge receipt below owns row accounting. Do not reuse an
# exported EXPECTED_ROWS=13553 from the old image+audio launcher.
unset H3_REF2AV_EXPECTED_ROWS
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
export H3_REF2AV_DMD_OUTPUT="${H3_REF2AV_DMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_100k}"
export H3_RDZV_ID="${H3_RDZV_ID:-h3_ref2av_omni_imageonly_dmd8_legacy_shift12_100k}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi
: "${MASTER_ADDR:?ACP must set a common MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set a common MASTER_PORT on all four workers}"

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
if training["method"] != "dmd" or matching.get("projected_dmd", False) or dmd.get("residual_head", {}).get("enabled", False):
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
    f"H3 Omni legacy-shift12: plain DMD, steps={dmd['num_inference_steps']}, "
    f"iters={training['max_train_iters']}, grad_accum={training['gradient_accumulation_iters']}, "
    f"legacy_numerics={matching.get('legacy_numerics', False)}, "
    f"model_mode={dmd.get('model_mode', 'eval')}, update_order={dmd.get('update_order', 'student_first')}, "
    f"fake_update_ratio={dmd['fake_update_ratio']}, CFG={training['teacher']['guidance_scale']}"
)
print(f"config={sys.argv[1]}\ndata={manifest}\noutput={training['output_dir']}")
print("legacy_numerics aligns condition noise and sigma arithmetic; x0 reconstruction remains FP32.")
print(f"sampler={json.dumps(sampler, sort_keys=True)}; workers={data['num_workers']}, pin_memory={data['pin_memory']}")
print(
    "Default sampling: one of the observed 1..6 image counts per 32-row global microbatch; "
    "16 landscape + 16 portrait. "
    "Balance observed image counts over the data epoch, rotating the omitted majority rows; "
    "NOT a full-data pass and NOT mixed-count coverage per batch. "
    "Each observed count/orientation cell needs at least 16 rows."
)
for name in ("student", "fake", "teacher"):
    effective = model if name == "student" else {**model, **model.get(name, {})}
    print(f"precision {name}: transformer_param_dtype={effective['transformer_param_dtype']}, running_dtype={effective['running_dtype']}")
for name in ("student", "fake"):
    print(f"{name}: {json.dumps(training[name], sort_keys=True)}")
print("Use a NEW output directory; old-trainer/PDMD/HEAD checkpoints are not a fresh aligned DMD run.")
PY

# Shared launcher owns FSDP32 torchrun, model/kernel checks and --dry-run.
exec bash "$H3_CODE_ROOT/lightx2v_train/scripts/run_minimax_h3_ref2av_fsdp32_32gpu_acp.sh" "$@"
