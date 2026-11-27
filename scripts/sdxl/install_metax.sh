#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
"${python_bin}" -c 'import torch; assert "metax" in torch.__version__, "Use the MetaX PyTorch environment"' 
"${python_bin}" -m pip install setuptools wheel packaging ninja
"${python_bin}" -m pip install -r "${root}/scripts/sdxl/requirements-metax.txt"
