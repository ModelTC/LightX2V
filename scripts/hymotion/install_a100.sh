#!/bin/bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
"${python_bin}" -m pip install 'setuptools>=61' wheel packaging ninja
"${python_bin}" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"${python_bin}" -m pip install -r "${root}/scripts/hymotion/requirements-a100.txt"
echo "Core environment ready. See scripts/hymotion/README.md for model preparation."
