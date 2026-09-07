#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command="prepare"

if [[ "${1:-}" == "check" || "${1:-}" == "--check" ]]; then
  command="check"
  shift
elif [[ "${1:-}" == "setup" || "${1:-}" == "prepare" ]]; then
  shift
fi

exec python "${script_dir}/support/runtime.py" "${command}" "$@"
