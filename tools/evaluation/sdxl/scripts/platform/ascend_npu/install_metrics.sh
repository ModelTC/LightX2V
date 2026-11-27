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
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
"$PYTHON_BIN" -m pip download --no-deps --no-binary=:all: --no-build-isolation mmcv-full==1.7.2 -d "$TEMP"
echo 'aa276a2c68fa84db9bbcdcf6ae941e30f0cb4a7f175bd283516669e00bab3759  mmcv-full-1.7.2.tar.gz' > "$TEMP/SHA256SUMS"
(cd "$TEMP" && sha256sum -c SHA256SUMS)
tar --no-same-owner -xzf "$TEMP/mmcv-full-1.7.2.tar.gz" -C "$TEMP"
(cd "$TEMP/mmcv-full-1.7.2" && MMCV_WITH_OPS=1 MAX_JOBS="${MAX_JOBS:-4}" "$PYTHON_BIN" "$HERE/scripts/platform/ascend_npu/build_cpu_extension.py")
"$PYTHON_BIN" -m pip install --no-deps "$TEMP"/mmcv-full-1.7.2/dist/*.whl
