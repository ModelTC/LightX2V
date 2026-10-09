#!/usr/bin/env bash
set -Eeuo pipefail

# ACP: run this identical command on each of four workers, eight GPUs each.
# MASTER_ADDR / MASTER_PORT must refer to the same reachable rendezvous host.
# Keep the existing filename/entry point; the selected recipe is now PDMD.
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_CONFIG_PATH="${H3_CONFIG_PATH:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_match124_bf16_fsdp32.yaml}"
: "${MASTER_ADDR:?ACP must set a common MASTER_ADDR on all four workers}"
: "${MASTER_PORT:?ACP must set a common MASTER_PORT on all four workers}"
export H3_MODEL_PATH="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
export H3_REF2AV_CACHE="${H3_REF2AV_CACHE:-/mnt/lm_data_afs/gushiqiao/datasets/omni_r2v_image_only_100k_20261004/latent_match124_bf16/metadata.jsonl}"
export H3_REF2AV_DMD_OUTPUT="${H3_REF2AV_DMD_OUTPUT:-$H3_CODE_ROOT/outputs/minimax_h3_ref2av_omni_imageonly_pdmd8_paper_10k}"
export H3_PDMD="${H3_PDMD:-true}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi
if [[ "$H3_PDMD" != true ]]; then
    echo "This is the H3 paper-hyperparameter PDMD recipe; unset H3_PDMD or set it to true." >&2
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
model = config["model"]
mixed = distributed["fsdp2"]["mixed_precision"]
roles = (training["student"], training["fake"])
batch_mode = sampler.get("batch_mode")
if batch_mode == "count_coverage":
    sampling_checks = {
        "no category/orientation undersampling": sampler.get("balance_image_counts") is False and sampler.get("balance_orientation") is False,
    }
elif batch_mode == "count_random":
    sampling_checks = {
        "balanced image counts 1..6 with random orientations": sampler.get("image_counts") == list(range(1, 7)) and sampler.get("balance_image_counts") is True and sampler.get("balance_orientation") is False,
        "rotating subsets for random single-count batches": sampler.get("strict_full_epoch") is False and sampler.get("remainder_policy") == "rotating_drop",
    }
else:
    raise SystemExit("Omni image-only PDMD recipe requires batch_mode=count_coverage or count_random.")

checks = {
    "PDMD projection enabled": training["method"] == "dmd" and matching.get("projected_dmd") is True,
    "8 steps, 10000 iterations, grad accumulation 1": dmd["num_inference_steps"] == 8 and training["max_train_iters"] == 10000 and training["gradient_accumulation_iters"] == 1,
    "student-first, 5 critic updates": dmd["update_order"] == "student_first" and dmd["fake_update_ratio"] == 5,
    "FSDP32, no sequence parallel": distributed["fsdp2"]["enabled"] and distributed["fsdp2"]["size"] == 32 and not distributed["sequence_parallel"]["enabled"],
    "per-rank microbatch 1 with image-only references": data["batch_size"] == 1 and sampler.get("require_image_only") is True,
    **sampling_checks,
    "metadata-controlled 124-frame generation": matching.get("geometry_from_metadata") is True and dmd["generation_shapes"] == [{"value": [124, 768, 1344]}],
    "H3 paper student/critic learning rates 5e-5 / 1e-5": roles[0]["optimizer"]["learning_rate"] == 5e-5 and roles[1]["optimizer"]["learning_rate"] == 1e-5,
    "H3 paper AdamW betas (0, 0.9), no weight decay": all(role["optimizer"]["adam_beta1"] == 0.0 and role["optimizer"]["adam_beta2"] == 0.9 and role["optimizer"]["weight_decay"] == 0.0 for role in roles),
    "H3 paper video/audio shifts 12/3 and no CFG": matching["video_flow_shift"] == 12.0 and matching["audio_flow_shift"] == 3.0 and training["teacher"]["guidance_scale"] == 1.0,
    "8 DataLoader workers per rank with pinned memory": data["num_workers"] == 8 and data["pin_memory"] is True,
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
print(f"H3 Omni Ref2AV: 4 x 8 GPUs, PDMD, steps=8, iters=10000, grad_accum=1, student_first, fake_update_ratio=5, global_microbatch=32, batch_mode={batch_mode}, output={training['output_dir']}")
print(f"config={sys.argv[1]}")
if batch_mode == "count_random":
    print("Sampling: one image count per global batch; observed counts 1..6 balanced across each sampler epoch; random orientations with no fixed quota. Majority-count subsets rotate across epochs, NOT a full-data pass per epoch. Each observed image count needs at least 32 rows total, not 16 rows per orientation.")
for name, role in zip(("student", "critic"), roles):
    details = f"train_type={role['train_type']}"
    if role["train_type"] == "lora":
        lora = role.get("lora", {})
        details += f", rank={lora.get('rank')}, alpha={lora.get('alpha')}"
    print(f"{name}: {details}")
for name in ("student", "fake", "teacher"):
    # Report configured precision without imposing a fixed recipe. Only fake
    # and teacher support role-local overrides, matching the training runtime.
    override = {} if name == "student" else model.get(name, {})
    effective = {**model, **override}
    role_mixed = override.get("distributed", {}).get("fsdp2", {}).get("mixed_precision", {})
    effective_mixed = {**mixed, **role_mixed}
    print(
        f"precision {name}: transformer_param_dtype={effective.get('transformer_param_dtype')}, "
        f"running_dtype={effective.get('running_dtype')}, use_autocast={effective.get('use_autocast')}, "
        f"fsdp={json.dumps(effective_mixed, sort_keys=True)}"
    )
print("H3 paper optimizer/TTUR settings; train types and LoRA settings are read from config. Retained user overrides: Ref2AV, 8 NFE, 768p, global batch 32. Each local outer iter counts 1 student + 5 critic updates; 10000 outer iters = 60000 optimizer updates, not 10000 paper iterations. Use a new output directory when changing the algorithm, train types, or LoRA settings.")
PY

cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --nnodes=4 --nproc_per_node=8
    "--rdzv_id=${H3_RDZV_ID:-h3_ref2av_omni_imageonly_pdmd8_paper_10k}"
    --rdzv_backend=c10d "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    --max_restarts=0 train.py --config "$H3_CONFIG_PATH"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
