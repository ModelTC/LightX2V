#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: bash lightx2v_train/scripts/openpi/run_pi05_finetune_ema.sh" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset OPENPI_RESUME_CHECKPOINT
exec bash "${script_dir}/support/launch_pi05_libero.sh"
