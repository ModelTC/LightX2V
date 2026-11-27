#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
"${python_bin}" -c 'import torch; import torch_npu; assert torch.npu.is_available(), "Use an Ascend PyTorch environment"; assert torch.npu.is_bf16_supported(), "Native BF16 support is required"'
"${python_bin}" -m pip install -r "${root}/scripts/hymotion/requirements-ascend.txt"
