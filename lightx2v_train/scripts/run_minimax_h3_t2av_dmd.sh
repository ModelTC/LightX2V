#!/usr/bin/env bash

set -euo pipefail

H3_CODE_ROOT="${H3_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
H3_PYTHON="${H3_PYTHON:-python}"
H3_CONFIG_PATH="${H3_CONFIG_PATH:-$H3_CODE_ROOT/lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora.yaml}"
# The supplied config shards student, fake, and teacher across all eight GPUs.
# Override this when launching on a different node shape.
H3_NUM_PROCESSES="${H3_NUM_PROCESSES:-8}"
export H3_PDMD="${H3_PDMD:-false}"
export PYTHONPATH="$H3_CODE_ROOT/lightx2v_train${PYTHONPATH:+:$PYTHONPATH}"
case "$H3_PDMD" in
    true|false) ;;
    *) echo "H3_PDMD must be true or false." >&2; exit 1 ;;
esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 1
fi
[[ -f "$H3_CONFIG_PATH" ]] || { echo "Missing training config: $H3_CONFIG_PATH" >&2; exit 1; }
H3_CONFIG_PATH="$(cd "$(dirname "$H3_CONFIG_PATH")" && pwd)/$(basename "$H3_CONFIG_PATH")"

"$H3_PYTHON" - "$H3_CONFIG_PATH" <<'PY'
import os
import sys
from lightx2v_train.runtime.config import load_config

config = load_config(sys.argv[1])
projected = config["model"].get("capabilities", {}).get("distribution_matching", {}).get("projected_dmd", False)
expected = os.environ["H3_PDMD"] == "true"
dmd = config["training"]["dmd"]
order = dmd.get("update_order", "student_first")
ratio = int(dmd["fake_update_ratio"])
if projected is not expected:
    raise SystemExit("H3_PDMD disagrees with the selected config's projected_dmd.")
if expected and (order != "student_first" or ratio != 1):
    raise SystemExit("H3 PDMD requires update_order=student_first and fake_update_ratio=1.")
print(f"H3 T2AV: PDMD={str(projected).lower()}, update_order={order}, fake_update_ratio={ratio}, output={config['training']['output_dir']}")
PY

cd "$H3_CODE_ROOT/lightx2v_train"
command=(
    "$H3_PYTHON" -u -m torch.distributed.run
    --standalone
    "--nproc_per_node=${H3_NUM_PROCESSES}"
    train.py --config "$H3_CONFIG_PATH"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
