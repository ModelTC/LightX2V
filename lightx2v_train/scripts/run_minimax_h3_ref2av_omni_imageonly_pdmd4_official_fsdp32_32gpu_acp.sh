#!/usr/bin/env bash
set -Eeuo pipefail

# Run the same command on all four ACP workers (eight GPUs each).
# Independent source-aligned PDMD entry point: stale H3_CONFIG_PATH/H3_PDMD
# variables must not select one of the old outer-iteration training recipes.
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_PDMD_OFFICIAL_CONFIG="${H3_PDMD_OFFICIAL_CONFIG:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_omni_imageonly_pdmd4_official_fsdp32.yaml}"
: "${MASTER_ADDR:?ACP must set a common MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set a common MASTER_PORT on all four workers}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
export H3_REF2AV_DMD_OUTPUT="${H3_REF2AV_DMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_ref2av_omni_imageonly_pdmd4_official_lorafp32_250000updates}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi
command -v "$H3_PYTHON" >/dev/null
if [[ ! -f "$H3_PDMD_OFFICIAL_CONFIG" || ! -s "$H3_REF2AV_CACHE" ]]; then
    echo "Training config and nonempty merged Ref2AV cache manifest must exist." >&2
    exit 1
fi
H3_PDMD_OFFICIAL_CONFIG="$(cd "$(dirname "$H3_PDMD_OFFICIAL_CONFIG")" && pwd)/$(basename "$H3_PDMD_OFFICIAL_CONFIG")"

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
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3_pdmd_official_pycache}"

# Read-only CPU preflight. Do not import torch or deserialize latent payloads.
"$H3_PYTHON" - "$H3_PDMD_OFFICIAL_CONFIG" <<'PY'
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

config_module = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) / "lightx2v_train/runtime/config.py"
spec = importlib.util.spec_from_file_location("_h3_pdmd_official_config", config_module)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.load_config(sys.argv[1])
training, model = config["training"], config["model"]
dmd, data = training["dmd"], config["data"]["train"]
distributed = config["distributed"]
matching = model["capabilities"]["distribution_matching"]
sampler = data["reference_cost_sampler"]
checks = {
    "the official PDMD training flow and losses": training["method"] == "dmd" and dmd.get("official_pdmd") is True and matching.get("official_pdmd") is True and matching.get("projected_dmd") is True,
    "4-step fake-first, 5 critic updates per student update": dmd["num_inference_steps"] == 4 and dmd["update_order"] == "fake_first" and dmd["fake_update_ratio"] == 5,
    "positive single-optimizer-update iteration count": type(training["max_train_iters"]) is int and training["max_train_iters"] > 0,
    "FSDP32 without sequence parallel": distributed["fsdp2"]["enabled"] and distributed["fsdp2"]["size"] == 32 and not distributed["sequence_parallel"]["enabled"] and distributed["sequence_parallel"]["size"] == 1,
    "global microbatch 32 (per-rank batch 1, accumulation 1)": data["batch_size"] == 1 and training["gradient_accumulation_iters"] == 1,
    "the existing Omni Ref image-only latent dataset": model["name"] == "minimax_h3_ref2av" and data["name"] == "minimax_h3_ref_cache_dataset" and sampler.get("require_image_only") is True,
    "balanced image counts 1..6 and random orientations": sampler.get("batch_mode") == "count_random" and sampler.get("image_counts") == list(range(1, 7)) and sampler.get("balance_image_counts") is True and sampler.get("balance_orientation") is False,
    "rotating image-count subsets": sampler.get("strict_full_epoch") is False and sampler.get("remainder_policy") == "rotating_drop",
    "metadata-controlled 124-frame generation": matching.get("geometry_from_metadata") is True and dmd["generation_shapes"] == [{"value": [124, 768, 1344]}],
}
for description, passed in checks.items():
    if not passed:
        raise SystemExit(f"Official PDMD4 launcher requires {description}.")
model_path = Path(model["pretrained_model_name_or_path"])
if not (model_path / "transformer_ref/config.json").is_file():
    raise SystemExit(f"Missing Ref2AV transformer config: {model_path / 'transformer_ref/config.json'}")
manifest = Path(data["data_path"])
for name, path in (("H3_MODEL_PATH", model_path), ("H3_REF2AV_CACHE", manifest), ("H3_REF2AV_DMD_OUTPUT", Path(training["output_dir"]))):
    if path.resolve() != Path(os.environ[name]).resolve():
        raise SystemExit(f"Resolved config disagrees with {name}: {path}")
receipt_path = manifest.with_suffix(".complete.json")
if not receipt_path.is_file():
    raise SystemExit(f"Missing completed cache merge receipt: {receipt_path}; finish the checked shard merge first.")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
counts = [receipt.get(name) for name in ("completed_count", "failed_count", "input_total_rows")]
if any(type(value) is not int or value < 0 for value in counts) or counts[0] == 0 or counts[0] + counts[1] != counts[2]:
    raise SystemExit(f"Invalid completed/failed/source accounting in {receipt_path}.")
if not receipt.get("preprocess_fingerprint"):
    raise SystemExit(f"Missing preprocessing fingerprint in {receipt_path}.")
digest, rows = hashlib.sha256(), 0
with manifest.open("rb") as handle:
    for line in handle:
        digest.update(line)
        rows += bool(line.strip())
if digest.hexdigest() != receipt.get("manifest_sha256") or rows != counts[0]:
    raise SystemExit("Cache manifest no longer matches its complete receipt; re-run the checked shard merge.")
student_updates = training["max_train_iters"] // (dmd["fake_update_ratio"] + 1)
critic_updates = training["max_train_iters"] - student_updates
print(f"Verified cache: completed={counts[0]}, failed={counts[1]}, input_rows={counts[2]}, ref_image_counts={receipt.get('reference_image_counts', {})}")
print(f"Official PDMD Ref2AV: 4 x 8 GPUs, steps=4, optimizer_updates={training['max_train_iters']}, student_updates={student_updates}, critic_updates={critic_updates}, fake_first, global_microbatch=32, batch_mode=count_random")
print(f"Checkpoint: save_every_iters={training['save_every_iters']}, save_total_limit={training['save_total_limit']}")
print("Each iteration is ONE role's optimizer update, not a student+critic outer round. Critic uses a full rollout; student reuses the last critic rollout.")
print("Sampling: one image count per global batch; balanced counts 1..6; natural random orientations, no fixed 16+16 quota. Rotating subsets are NOT a full-data pass per epoch.")
print(f"config={sys.argv[1]}\ndata={manifest}\noutput={training['output_dir']}")
for role in ("student", "fake"):
    settings = training[role]
    details = f"train_type={settings['train_type']}, lr={settings['optimizer']['learning_rate']}"
    if settings["train_type"] == "lora":
        details += f", rank={settings['lora']['rank']}, alpha={settings['lora']['alpha']}"
        details += f", lora_param_dtype={settings['lora'].get('param_dtype', 'inherit_base')}"
    print(f"{role}: {details}")
for role in ("student", "fake", "teacher"):
    override = {} if role == "student" else model.get(role, {})
    effective = {**model, **override}
    mixed = {**distributed["fsdp2"]["mixed_precision"], **override.get("distributed", {}).get("fsdp2", {}).get("mixed_precision", {})}
    print(f"precision {role}: transformer_param_dtype={effective.get('transformer_param_dtype')}, running_dtype={effective.get('running_dtype')}, use_autocast={effective.get('use_autocast')}, fsdp={json.dumps(mixed, sort_keys=True)}")
print("Retained task overrides: Ref2AV image-only cache, 768p, batch 32, student LoRA rank128/alpha8 with configurable adapter parameter dtype, full fake, existing base precision/FSDP2. Use a NEW output directory; old outer-iteration checkpoints are not compatible.")
PY

cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=4 --nproc_per_node=8
    "--rdzv_id=${H3_RDZV_ID:-h3_ref2av_omni_imageonly_pdmd4_official_lorafp32_250000updates}"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$H3_PDMD_OFFICIAL_CONFIG"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
