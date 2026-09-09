#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash lightx2v_train/scripts/openpi/run_pi05_resume_ema.sh CHECKPOINT" >&2
  exit 2
fi

checkpoint="$1"
if [[ ! -d "${checkpoint}" ]]; then
  echo "Checkpoint directory not found: ${checkpoint}" >&2
  exit 1
fi
checkpoint="$(cd -- "${checkpoint}" && pwd -P)"

required_files=(
  model.safetensors
  ema/model.safetensors
  training_state.pt
  manifest.json
  _SUCCESS
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${checkpoint}/${path}" ]]; then
    echo "Not a complete LightX2V training checkpoint; missing ${checkpoint}/${path}" >&2
    exit 1
  fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENPI_RESUME_CHECKPOINT="${checkpoint}"
export OPENPI_TRAIN_OUTPUT="${OPENPI_TRAIN_OUTPUT:-$(dirname -- "${checkpoint}")}"
exec bash "${script_dir}/support/launch_pi05_libero.sh"
