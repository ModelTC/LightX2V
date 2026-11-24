#!/bin/bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
"${python_bin}" -m pip install 'setuptools>=61' wheel packaging ninja
"${python_bin}" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"${python_bin}" -m pip install -r "${root}/scripts/three_models/requirements-a100.txt"
# Keep the parent repository on PYTHONPATH; its optional training, service and
# video dependencies need not be installed for these three inference runners.
echo "Core environment ready. Build SF3D geometry extensions as described in scripts/sf3d/README.md."
