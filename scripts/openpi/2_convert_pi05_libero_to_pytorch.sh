#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
openpi_root="${OPENPI_PATH:-$(cd -- "${script_dir}/../../.." && pwd)/openpi}"
convert_python="${OPENPI_CONVERT_PYTHON:-${openpi_root}/.venv/bin/python}"

exec "${convert_python}" "${script_dir}/support/convert_jax_checkpoint.py" "$@"
