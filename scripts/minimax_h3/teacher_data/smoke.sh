#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${H3_PYTHON:-python}" -u "$script_dir/launch.py" smoke "$@"
