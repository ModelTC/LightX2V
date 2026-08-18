#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$project_dir"

: "${PYTHON:=python3}"
: "${CMAKE:=cmake}"
: "${CXX:=icpx}"
: "${XPU_TARGET:=bmg}"

command -v "$PYTHON" >/dev/null || { echo "ERROR: Python not found: $PYTHON" >&2; exit 1; }
command -v "$CMAKE" >/dev/null || { echo "ERROR: CMake not found: $CMAKE" >&2; exit 1; }
command -v ninja >/dev/null || { echo "ERROR: ninja not found" >&2; exit 1; }
command -v "$CXX" >/dev/null || { echo "ERROR: oneAPI C++ compiler not found: $CXX" >&2; exit 1; }
# Keep the selected interpreter stable inside pip/scikit-build subprocesses,
# where PATH may resolve a bare "python3" to the system installation.
PYTHON=$($PYTHON -c 'import sys; print(sys.executable)')
"$PYTHON" -c 'import scikit_build_core' >/dev/null 2>&1 || {
    echo "ERROR: scikit-build-core is not installed for $PYTHON" >&2
    echo "Install it with: $PYTHON -m pip install scikit-build-core wheel" >&2
    exit 1
}

torch_root=$($PYTHON -c 'import os, torch; print(os.path.dirname(torch.__file__))')
python_include=$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
if [[ ! -f "$python_include/Python.h" ]]; then
    echo "ERROR: Python development headers not found: $python_include/Python.h" >&2
    echo "Install the development package matching $($PYTHON --version 2>&1)." >&2
    exit 1
fi

echo "=== Step 1: Build ESIMD shared library ==="
CXX="$CXX" "$project_dir/lgrf_uni/build.sh"

echo "=== Step 2: Build Python extension ==="
"$CMAKE" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_PREFIX_PATH="$torch_root" \
    -DPython_EXECUTABLE="$PYTHON" \
    -DPython_INCLUDE_DIR="$python_include" \
    -DENABLE_CUTE_FMHA=ON \
    -DXPU_TARGET="$XPU_TARGET" \
    -B _cmake_build -S .
"$CMAKE" --build _cmake_build --parallel

echo "=== Step 3: Copy artifacts ==="
mkdir -p python/sycl_kernels
find _cmake_build -maxdepth 1 -type f -name '_ext*.so' -exec cp -f {} python/sycl_kernels/ \;
find _cmake_build -maxdepth 1 -type f -name 'rms_norm_torch*.so' -exec cp -f {} python/sycl_kernels/ \;
find _cmake_build -maxdepth 1 -type f -name 'cute_fmha_torch*.so' -exec cp -f {} python/sycl_kernels/ \;
cp -f lgrf_uni/libesimd.unify.lgrf.so python/sycl_kernels/

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    echo "=== Step 4: Smoke test ==="
    PYTHONPATH="$project_dir/python${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
        'import sycl_kernels; assert sycl_kernels.has_cute_fmha()'
fi

echo "=== Step 5: Build wheel ==="
mkdir -p dist
CMAKE_ARGS="-DCMAKE_CXX_COMPILER=$CXX -DPython_EXECUTABLE=$PYTHON -DPython_INCLUDE_DIR=$python_include -DENABLE_CUTE_FMHA=ON -DXPU_TARGET=$XPU_TARGET" \
    "$PYTHON" -m pip wheel . \
    --no-build-isolation --no-deps -w dist

echo "Build complete. Wheels are in $project_dir/dist"
