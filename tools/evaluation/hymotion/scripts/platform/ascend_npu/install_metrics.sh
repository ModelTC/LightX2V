#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEMP="$(mktemp -d)"
trap 'rm -rf -- "$TEMP"' EXIT
# Use a dedicated environment based on a provisioned matching torch/torch_npu
# pair. Never allow the metric dependencies to replace the platform runtime.
"$PYTHON_BIN" - "$TEMP/constraints.txt" <<'PY'
import importlib.metadata
import pathlib
import sys
import torch
import torch_npu
lines = ["torch==" + torch.__version__, "torch-npu==" + importlib.metadata.version("torch-npu")]
try:
    lines.append("torchvision==" + importlib.metadata.version("torchvision"))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit("Install torchvision matching the provisioned torch version first")
pathlib.Path(sys.argv[1]).write_text("\n".join(lines) + "\n")
PY
"$PYTHON_BIN" -m pip install -c "$TEMP/constraints.txt" -r "$HERE/configs/ascend_npu/requirements.txt"
