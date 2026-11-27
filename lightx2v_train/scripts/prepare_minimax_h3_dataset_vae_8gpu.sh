#!/usr/bin/env bash
# Restore API teachers, then encode missing AV latents without teacher sampling.
set -Eeuo pipefail

train_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${H3_PYTHON:-python}"
data_root="${H3_DATA_ROOT:-/mnt/miaohua/gushiqiao/datasets}"
dataset="$data_root/fl2av_dataset/h3_self_gen"
model_path="${H3_MODEL_PATH:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3}"
state="$dataset/.pipeline"
logs="$dataset/logs"
mkdir -p "$state" "$logs"
exec 9>"$state/lock"
flock -n 9 || { echo 'Stop the existing dataset preparation pipeline first.' >&2; exit 1; }
trap 'echo "Pipeline stopped at line $LINENO; rerun this command to resume." >&2' ERR
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "$state/api_video_vae_v1.organized" ]]; then
    echo '[1/2] Restore existing API videos, including original-prompt requests'
    "$python_bin" - "$dataset" <<'PY'
from pathlib import Path
import shutil
import sys
root=Path(sys.argv[1])
for name in ('dataset.json','report.json'):
    source=root/name
    backup=root/(source.stem+'.before_api_video_vae.json')
    if source.exists() and not backup.exists():
        shutil.copy2(source,backup)
PY
    if [[ -f "$data_root/h3_teacher_i2av_images_9458.zip" ]]; then
        unzip -n "$data_root/h3_teacher_i2av_images_9458.zip" -d "$data_root" > "$logs/extract_images.log" 2>&1
    fi
    extra=()
    if [[ -f "$data_root/minimax_h3_teacher_latents_5s/metadata.jsonl" ]]; then
        extra+=(--teacher-latents "$data_root/minimax_h3_teacher_latents_5s")
    fi
    "$python_bin" -u "$train_root/data_process/minimax_h3/organize_h3_dataset.py" \
        --datasets-root "$data_root" --output "$dataset" \
        --extract-teacher-zip --check-latents "${extra[@]}" \
        2>&1 | tee "$logs/organize_api_video_vae.log"
else
    echo '[1/2] Resuming the API-video manifest and VAE receipts'
fi

"$python_bin" - "$dataset" <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
report=json.loads((root/'report.json').read_text())
if report['rejected_rows']:
    raise ValueError(f"Organizer rejected {len(report['rejected_rows'])} rows; inspect {root/'report.json'}")
rows=json.loads((root/'dataset.json').read_text())
if not rows:
    raise ValueError('Dataset is empty')
for row in rows:
    for key in ('condition','image','teacher_video','teacher_video_latent','teacher_audio_latent'):
        if row.get(key) is not None:
            path=Path(row[key])
            if not path.is_absolute() or not path.is_file() or root not in path.resolve().parents:
                raise ValueError(f"{row['sample_id']}: invalid organized asset {key}: {path}")
print(f"Records: {len(rows)}; existing videos: {sum(bool(r.get('teacher_video')) for r in rows)}; missing videos left null: {sum(not r.get('teacher_video') for r in rows)}",flush=True)
PY
touch "$state/api_video_vae_v1.organized"

echo '[2/2] VAE-only encoding: existing video/audio latents reused; no DiT or condition encoding'
"$python_bin" -u "$train_root/data_process/minimax_h3/complete_h3_dataset.py" \
    --manifest "$dataset/dataset.json" --model-path "$model_path" \
    --gpus "${H3_GPUS:-0,1,2,3,4,5,6,7}" --encode-only \
    2>&1 | tee -a "$logs/vae_completion.log"
echo "Complete: $dataset/dataset.json"
echo "Missing video records: $dataset/vae_completion_report.json"
