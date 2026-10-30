#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi
H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_TRAINING_ADAPTER_DIR="${H3_TRAINING_ADAPTER_DIR:-/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-TrainingAdapter}"
command=("$H3_PYTHON" -u "$H3_CODE_ROOT/lightx2v_train/scripts/download_minimax_h3_training_adapter.py"
    --output-dir "$H3_TRAINING_ADAPTER_DIR" --revision "${H3_TRAINING_ADAPTER_REVISION:-master}"
    --download-workers "${H3_TRAINING_ADAPTER_DOWNLOAD_WORKERS:-4}")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
command -v "$H3_PYTHON" >/dev/null
mkdir -p "$H3_TRAINING_ADAPTER_DIR"
command -v flock >/dev/null
exec 9>"$H3_TRAINING_ADAPTER_DIR/.download.lock"
flock -n 9 || { echo "Another adapter downloader owns $H3_TRAINING_ADAPTER_DIR" >&2; exit 1; }
if ! "$H3_PYTHON" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("modelscope") is None)'; then
    "$H3_PYTHON" -m pip install modelscope
fi
"${command[@]}" 2>&1 | tee -a "$H3_TRAINING_ADAPTER_DIR/download.log"
