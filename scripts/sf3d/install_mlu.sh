#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
"${python_bin}" -c 'import torch, torch_mlu; assert hasattr(torch, "mlu"), "Use the torch_mlu environment"'
"${python_bin}" -m pip install setuptools wheel packaging ninja
"${python_bin}" -m pip install -r "${root}/scripts/sf3d/requirements-mlu.txt"

# Use the released CPU extensions for complete UV and PBR texture output.
if ! "${python_bin}" -c 'from texture_baker import TextureBaker; from uv_unwrapper import Unwrapper'; then
    extension_source="${SF3D_EXTENSION_SOURCE:-}"
    build_root=""
    if [[ -z "${extension_source}" ]]; then
        build_root="$(mktemp -d)"
        trap 'rm -rf -- "${build_root}"' EXIT
        git init --quiet "${build_root}"
        git -C "${build_root}" remote add origin https://github.com/Stability-AI/stable-fast-3d.git
        GIT_LFS_SKIP_SMUDGE=1 git -C "${build_root}" fetch --depth=1 origin ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2
        GIT_LFS_SKIP_SMUDGE=1 git -C "${build_root}" checkout --detach FETCH_HEAD
        extension_source="${build_root}"
    fi
    for extension in texture_baker uv_unwrapper; do
        (
            cd -- "${extension_source}/${extension}"
            USE_CUDA=0 MAX_JOBS="${MAX_JOBS:-2}" "${python_bin}" "${root}/scripts/sf3d/build_cpu_extension.py"
            "${python_bin}" -m pip install --no-deps dist/*.whl
        )
    done
fi
