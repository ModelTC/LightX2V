#!/usr/bin/env bash
# Launch three independent, fresh two-GPU runs. Never stop other GPU users.
set -Eeuo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash run_wan21_dmd_pdmd_head_10k_fsdp2.sh [--dry-run|--preflight]' \
        '  --dry-run: print the plan only; no GPU checks, writes or training.' \
        '  --preflight: verify inputs, training Python and idle GPUs; do not launch.' \
        'Environment: WAN_DMD_PYTHON (python), WAN_DMD_MODEL, WAN_DMD_PROMPTS,' \
        '  WAN_DMD_RUN_ROOT (new timestamp directory), DMD_GPUS (0,1),' \
        '  PDMD_GPUS (2,3), HEAD_GPUS (5,6).' \
        'The 10k config disables auto-resume and keeps one checkpoint per group.'
}
mode=run
if (($# > 1)); then usage >&2; exit 2; fi
case "${1:-}" in
    '') ;;
    --dry-run) mode=dry-run ;;
    --preflight) mode=preflight ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

train_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd "$train_root/.." && pwd)
config="$train_root/configs/train/dmd/wan2_1_t2v_1_3b_head_comparison_10k_fsdp2.yaml"
export WAN_DMD_MODEL="${WAN_DMD_MODEL:-/data/nvme0/gushiqiao/models/official_models/Wan2.1-T2V-1.3B}"
export WAN_DMD_PROMPTS="${WAN_DMD_PROMPTS:-/data/nvme4/gushiqiao/new/Causal-Forcing/prompts/vidprom_filtered_extended.txt}"
export WAN_DMD_PYTHON="${WAN_DMD_PYTHON:-python}"
# POSIX UTC+8 works even in minimal CUDA images without zoneinfo/tzdata.
run_root="${WAN_DMD_RUN_ROOT:-$repo_root/outputs/wan21_dmd_pdmd_head_10k_acc4_calibrated_$(TZ=CST-8 date +%Y%m%d_%H%M%S)_$$}"
groups=(dmd pdmd head)
gpu_groups=("${DMD_GPUS:-0,1}" "${PDMD_GPUS:-2,3}" "${HEAD_GPUS:-5,6}")
declare -A selected_gpus=()
for gpu_group in "${gpu_groups[@]}"; do
    if [[ ! "$gpu_group" =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]]; then
        printf 'Each GPU group must contain exactly two numeric GPU indices: %s\n' "$gpu_group" >&2
        exit 2
    fi
    IFS=, read -r -a gpu_indices <<< "$gpu_group"
    for gpu in "${gpu_indices[@]}"; do
        if [[ -n "${selected_gpus[$gpu]:-}" ]]; then
            printf 'GPU %s appears more than once; groups must be disjoint.\n' "$gpu" >&2
            exit 2
        fi
        selected_gpus[$gpu]=1
    done
done
if [[ -e "$run_root" || -L "$run_root" ]]; then
    printf 'Refusing an existing run root (this is a fresh experiment): %s\n' "$run_root" >&2
    exit 2
fi

export PYTHONPATH="$train_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
printf 'Fresh 10000-iteration comparison; head: 5 fresh fit batches, accumulation 4+1, calibrated/independent gate.\n'
printf 'Run root: %s\n' "$run_root"
printf 'Model: %s\nPrompts: %s\n' "$WAN_DMD_MODEL" "$WAN_DMD_PROMPTS"
for index in 0 1 2; do
    printf '%s: GPUs=%s log=%s/%s/launch.log\n' \
        "${groups[$index]}" "${gpu_groups[$index]}" "$run_root" "${groups[$index]}"
done
if [[ "$mode" == dry-run ]]; then
    printf 'DRY RUN: nothing started; GPU availability and dependencies are NOT verified.\n'
    exit 0
fi

command -v "$WAN_DMD_PYTHON" >/dev/null || { printf 'Training Python not found: %s\n' "$WAN_DMD_PYTHON" >&2; exit 2; }
command -v setsid >/dev/null
command -v nvidia-smi >/dev/null
# A failed NVML query is fatal; never interpret unavailable monitoring as idle.
preflight_json=$("$WAN_DMD_PYTHON" - "$config" "$run_root" "${gpu_groups[@]}" <<'PY'
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

config_path, run_root = Path(sys.argv[1]), Path(sys.argv[2])
gpu_groups = dict(zip(("dmd", "pdmd", "head"), sys.argv[3:]))
selected = {index for value in gpu_groups.values() for index in value.split(",")}

def query(fields, kind="gpu"):
    result = subprocess.run(
        ["nvidia-smi", "--query-" + kind + "=" + fields,
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(result.stdout)) if row]

devices = {row[0]: row for row in query("index,uuid,memory.used,utilization.gpu")}
missing = selected - devices.keys()
if missing:
    raise SystemExit("Requested GPU indices are not visible: " + ",".join(sorted(missing)))
apps = query("gpu_uuid,pid,process_name", "compute-apps")
selected_uuids = {devices[index][1] for index in selected}
busy = [row for row in apps if row[0] in selected_uuids]
for index in sorted(selected, key=int):
    row = devices[index]
    try:
        memory, utilization = float(row[2]), float(row[3])
    except ValueError:
        raise SystemExit("Cannot establish idle GPU state: " + repr(row))
    if memory > 1024 or utilization > 5:
        busy.append(row)
if busy:
    raise SystemExit("Selected GPUs are busy; no jobs started and no processes killed: " + json.dumps(busy))

model = Path(os.environ["WAN_DMD_MODEL"])
for name in ("config.json", "diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
    path = model / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit("Missing model file: " + str(path))
prompts = Path(os.environ["WAN_DMD_PROMPTS"])
digest = hashlib.sha256()
prompt_count = 0
with prompts.open("rb") as handle:
    for line in handle:
        digest.update(line)
        prompt_count += bool(line.strip())
if prompt_count < 8:
    raise SystemExit("At least eight nonempty prompts are required")
parent = run_root.resolve().parent
while not parent.exists():
    parent = parent.parent
free_gib = shutil.disk_usage(parent).free / 2**30
if free_gib < 80:
    raise SystemExit("Need at least 80 GiB free for three checkpoint sets/previews; available %.1f GiB" % free_gib)

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(selected, key=int))
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 6:
    raise SystemExit("Training Python cannot access all six selected CUDA GPUs")
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.dmd.residual_head import ResidualHeadConfig

configs = {}
for group in gpu_groups:
    os.environ.update({"WAN_DMD_OUTPUT": str(run_root / group),
                       "WAN_DMD_PROJECTED": str(group == "pdmd").lower(),
                       "WAN_DMD_RESIDUAL_HEAD": str(group == "head").lower()})
    value = load_config(str(config_path))
    options = ResidualHeadConfig.from_mapping(value["training"]["dmd"]["residual_head"])
    assert value["training"]["max_train_iters"] == 10000 and not value["resume"]["auto_resume"]
    assert options.fit_grad_accum_steps == 4 and options.gate_mode == "calibrated"
    configs[group] = value
train_root = config_path.parents[3]
source_paths = [config_path, train_root / "scripts/run_wan21_dmd_pdmd_head_10k_fsdp2.sh"]
source_paths.extend(train_root / "lightx2v_train/trainers/dmd" / name for name in
                    ("residual_head.py", "residual_head_training.py", "runtime.py", "checkpoint.py"))
print(json.dumps({
    "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
    "torch": torch.__version__, "gpu_groups": gpu_groups,
    "gpu_snapshot": [devices[index] for index in sorted(selected, key=int)],
    "model": str(model.resolve()), "prompts": str(prompts.resolve()),
    "prompt_sha256": digest.hexdigest(), "prompt_count": prompt_count,
    "free_disk_gib": round(free_gib, 1), "configs": configs,
    "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
}, indent=2))
PY
)
if [[ "$mode" == preflight ]]; then
    printf '%s\n' "$preflight_json"
    printf 'Preflight passed. No output directory created and no training started.\n'
    exit 0
fi

mkdir -p "$(dirname "$run_root")"
mkdir "$run_root"
printf '%s\n' "$preflight_json" > "$run_root/run_manifest.json"
printf '%s\n' "$$" > "$run_root/launcher.pid"
cp -- "$config" "$run_root/config.yaml"
child_pids=()
stop_children() {
    trap - TERM INT
    for pid in "${child_pids[@]}"; do
        # Each child was launched under setsid. Signal only our still-live job.
        if kill -0 "$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    done
    wait || true
    exit 143
}
trap stop_children TERM INT
cd "$train_root"
for index in 0 1 2; do
    group="${groups[$index]}"
    projected=false; head=false
    [[ "$group" == pdmd ]] && projected=true
    [[ "$group" == head ]] && head=true
    mkdir "$run_root/$group"
    setsid env CUDA_VISIBLE_DEVICES="${gpu_groups[$index]}" \
        WAN_DMD_OUTPUT="$run_root/$group" WAN_DMD_PROJECTED="$projected" WAN_DMD_RESIDUAL_HEAD="$head" \
        "$WAN_DMD_PYTHON" -u -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
        "--rdzv_id=wan21_10k_${group}_$$" --max_restarts=0 \
        train.py --config "$config" > "$run_root/$group/launch.log" 2>&1 < /dev/null &
    pid=$!
    child_pids+=("$pid")
    printf '%s\n' "$pid" > "$run_root/$group/launcher.pid"
    printf 'START group=%s pid=%s GPUs=%s log=%s/%s/launch.log\n' \
        "$group" "$pid" "${gpu_groups[$index]}" "$run_root" "$group"
done
result=0
for index in 0 1 2; do
    rc=0
    wait "${child_pids[$index]}" || rc=$?
    printf '%s\n' "$rc" > "$run_root/${groups[$index]}/exit_code"
    printf 'END group=%s rc=%s\n' "${groups[$index]}" "$rc"
    if ((rc != 0)); then result=1; fi
done
trap - TERM INT
exit "$result"
