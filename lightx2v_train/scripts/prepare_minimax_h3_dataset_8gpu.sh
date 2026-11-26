#!/usr/bin/env bash
# Migrate once, organize once, then resume missing teacher/AV assets on eight GPUs.
set -Eeuo pipefail

train_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${H3_PYTHON:-python}"
old_root="${H3_OLD_DATA_ROOT:-/mnt/lm_data_afs/gushiqiao/datasets}"
data_root="${H3_DATA_ROOT:-/mnt/miaohua/gushiqiao/datasets}"
dataset="$data_root/fl2av_dataset/h3_self_gen"
teacher_cache="$data_root/minimax_h3_teacher_latents_5s"
model_path="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
kernel_path="${H3_FA3_KERNEL_PATH:-/mnt/lm_data_afs/gushiqiao/cache/kernels/kernels--kernels-community--flash-attn3/snapshots/43f0bd269777115d94ff826e0d113ce9c1c9087b}"
state="$dataset/.pipeline"
logs="$dataset/logs"
mkdir -p "$state" "$logs"
exec 9>"$state/lock"
flock -n 9 || { echo 'Another dataset preparation pipeline is running.' >&2; exit 1; }
trap 'echo "Pipeline stopped at line $LINENO; rerun this command to resume." >&2' ERR

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "$state/teacher_latents_copied" ]]; then
    echo "[1/3] Copy teacher cache to $teacher_cache"
    mkdir -p "$teacher_cache"
    rsync -aH --no-owner --no-group --partial --info=progress2 \
        "$old_root/minimax_h3_teacher_latents_5s/" "$teacher_cache/" \
        2>&1 | tee -a "$logs/copy_teacher_latents.log"
    touch "$state/teacher_latents_copied"
else
    echo '[1/3] Copy already completed; the old data directory is not accessed.'
fi

if [[ ! -f "$dataset/dataset.json" ]]; then
    echo '[2/3] Extract images and organize data inside miaohua'
    unzip -n "$data_root/h3_teacher_i2av_images_9458.zip" -d "$data_root" \
        > "$logs/extract_images.log" 2>&1
    "$python_bin" -u "$train_root/data_process/minimax_h3/organize_h3_dataset.py" \
        --datasets-root "$data_root" --output "$dataset" \
        --teacher-latents "$teacher_cache" --extract-teacher-zip --check-latents \
        2>&1 | tee "$logs/organize.log"
else
    echo '[2/3] Existing dataset.json retained, including completed teacher assets.'
fi

# Only metadata provenance may retain historical paths. Training assets must
# already reside in miaohua before the GPU stage starts.
"$python_bin" - "$dataset" "$data_root" <<'PY'
import json
from pathlib import Path
import sys

dataset, root = map(lambda value: Path(value).resolve(), sys.argv[1:])
report = json.loads((dataset / 'report.json').read_text())
if report['rejected_rows']:
    raise ValueError(f"Organizer rejected {len(report['rejected_rows'])} source rows; inspect {dataset / 'report.json'}")
rows = json.loads((dataset / 'dataset.json').read_text())
if not rows:
    raise ValueError('The organized dataset is empty')
for row in rows:
    for key in ('condition', 'image', 'teacher_video', 'teacher_video_latent', 'teacher_audio_latent'):
        value = row.get(key)
        if value is not None:
            path = Path(value)
            if not path.is_absolute() or not path.is_file() or root not in path.resolve().parents:
                raise ValueError(f"{row['sample_id']}: {key} must point to an existing file inside {root}: {value}")
print(f'Validated {len(rows)} records; all populated training asset paths are inside {root}', flush=True)
PY

echo '[3/3] Complete teacher videos and AV latents: 30 sigma points / 29 evaluations, shifts 12/3'
"$python_bin" -u "$train_root/data_process/minimax_h3/complete_h3_dataset.py" \
    --manifest "$dataset/dataset.json" --model-path "$model_path" \
    --condition-model-path "${H3_CONDITION_MODEL_PATH:-$model_path}" \
    --gpus "${H3_GPUS:-0,1,2,3,4,5,6,7}" --schedule-points 30 \
    --attention-kernel-path "$kernel_path" \
    2>&1 | tee -a "$logs/completion.log"

echo "Complete: $dataset/dataset.json"
