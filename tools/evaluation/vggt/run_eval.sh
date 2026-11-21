#!/bin/bash
set -euo pipefail
project_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${project_path}/evaluate.py" "$@"
