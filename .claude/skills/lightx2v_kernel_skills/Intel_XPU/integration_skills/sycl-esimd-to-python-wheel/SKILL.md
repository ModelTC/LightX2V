---
name: sycl-esimd-to-python-wheel
description: |
  Full pipeline for turning a SYCL/ESIMD GPU kernel into a Python-importable wheel package
  on Windows with Intel oneAPI 2025.x and conda. Covers every layer of the stack:
  ESIMD kernel (.cpp/.h) → Windows DLL (icpx) → PyTorch C++ extension (.pyd, CMake) →
  Python package → wheel (.whl, scikit-build-core).

  Use this skill whenever the user is working on Intel Arc GPU (Xe2 / BMG / PTL-H)
  SYCL or ESIMD kernels and wants to expose them to Python, package them as a wheel,
  set up a build script, debug build failures, or understand how the DLL + .pyd + wheel
  layers fit together. Also use it when they hit Windows-specific build issues like
  setvars.bat failing, cmake.exe producing no output, or ur_api.h not found.
---

# SYCL/ESIMD Kernel → Python Wheel (Windows, Intel oneAPI)

## Pipeline overview

```
lgrf_uni/kernels.cpp   ──icpx──►  my_kernel.dll + my_kernel.lib
                                         │
csrc/entry.cpp                           │ (linked via .lib)
csrc/wrapper.cpp       ──cmake──►  _ext.cp311-win_amd64.pyd
                                         │
python/my_package/                       │ (copied in)
  __init__.py          ──build──►  dist/my_package-0.0.1-cp311-abi3-win_amd64.whl
  _ext.*.pyd
  my_kernel.dll
```

Two compilation passes, always in this order:
1. **ESIMD DLL** — compiled with `icpx -fsycl`, AOT for the target GPU
2. **Python extension** — compiled with `icx` (host-only), links the DLL's `.lib`

---

## Canonical directory structure

```
my_kernel/
├── CMakeLists.txt
├── pyproject.toml
├── build.bat                     ← all-in-one build script
├── run_build.bat                 ← Claude Code launcher (sets conda env, calls build.bat)
├── lgrf_uni/
│   ├── esimd_kernel_api.h        ← dllexport / dllimport macro
│   ├── kernels.cpp               ← extern "C" dispatchers, sycl::queue interop
│   └── single_kernels/           ← header-only kernel implementations
│       └── my_kernel.h
├── csrc/
│   ├── entry.cpp                 ← PYBIND11_MODULE registrations
│   ├── my_kernel_wrapper.cpp     ← thin C++ wrapper (Tensor checks → DLL call)
│   └── utils.h                   ← get_queue() helper
├── python/my_package/
│   ├── __init__.py               ← DLL preload + _ext import
│   └── version.py
└── test/
    └── test_my_kernel.py
```

---

## Step 1 — Write the ESIMD kernel (DLL layer)

### `lgrf_uni/esimd_kernel_api.h`
```cpp
#pragma once
#ifdef BUILD_ESIMD_KERNEL_LIB
  #define ESIMD_KERNEL_API __declspec(dllexport)
#else
  #define ESIMD_KERNEL_API __declspec(dllimport)
#endif
```
Define `-DBUILD_ESIMD_KERNEL_LIB` **only** when compiling the DLL, not when linking it.

### `lgrf_uni/kernels.cpp`
```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "esimd_kernel_api.h"
#include "single_kernels/my_kernel.h"   // header-only ESIMD implementation

extern "C" ESIMD_KERNEL_API void my_kernel(
    void* input, void* output, int N,
    void* sycl_queue_ptr)               // pass PyTorch's queue as void*
{
    sycl::queue& q = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
    // launch via q.submit(...) using the header-only kernel
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(...), [=](sycl::nd_item<1> ndi)
            SYCL_ESIMD_KERNEL { my_kernel_impl(..., ndi); });
    }).wait();
}
```

Rules:
- Every exported function: `extern "C"` (no C++ name mangling) + `ESIMD_KERNEL_API`
- Accept `void* sycl_queue_ptr`, cast to `sycl::queue*` — this is how you share the queue with PyTorch
- Keep ESIMD kernel logic in header-only files under `single_kernels/`; `kernels.cpp` is just dispatching

### Build DLL command
```bat
icpx kernels.cpp -shared -o my_kernel.dll ^
    -DBUILD_ESIMD_KERNEL_LIB ^
    -fsycl -fsycl-targets=spir64_gen ^
    -Xs "-device ptl-h -options -doubleGRF" ^
    -O3
```
Output: `my_kernel.dll` (runtime) + `my_kernel.lib` (import library for the linker).

Device targets: `ptl-h` = Panther Lake, `xe2-hpg` = BMG. Use `-doubleGRF` for ESIMD kernels that need all 256 GRF registers.

---

## Step 2 — Write the PyTorch extension bridge (csrc layer)

### `csrc/utils.h` — borrow PyTorch's XPU queue
```cpp
#pragma once
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
namespace utils {
    static inline sycl::queue& get_queue(const torch::Device& device) {
        return c10::xpu::getCurrentXPUStream(device.index()).queue();
    }
}
```

### `csrc/my_kernel_wrapper.cpp`
```cpp
#include <torch/extension.h>
#include "../lgrf_uni/esimd_kernel_api.h"   // dllimport declarations
#include "utils.h"

// Forward-declare the DLL function (or include a header with the declaration)
extern "C" ESIMD_KERNEL_API void my_kernel(void*, void*, int, void*);

torch::Tensor my_kernel_torch(torch::Tensor input) {
    TORCH_CHECK(input.device().type() == c10::DeviceType::XPU, "input must be on XPU");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    auto output = torch::empty_like(input);
    sycl::queue& sq = utils::get_queue(input.device());
    my_kernel(input.data_ptr(), output.data_ptr(), (int)input.numel(), &sq);
    return output;
}
```

### `csrc/entry.cpp`
```cpp
#include <torch/extension.h>
torch::Tensor my_kernel_torch(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_kernel", &my_kernel_torch, "My ESIMD kernel", py::arg("input"));
}
```

---

## Step 3 — CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(my-kernels LANGUAGES CXX)

find_package(IntelSYCL REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# ── ur_api.h shim (required on Windows with conda + oneAPI 2025.x) ────────────
# conda ships a stale ur_api.h; shim points to the versioned oneAPI copy.
# FindIntelSYCL sets SYCL_INCLUDE_DIR = .../compiler/2025.x/include
# CORRECT:  ${SYCL_INCLUDE_DIR}/ur_api.h          (file is here)
# WRONG:    ${SYCL_INCLUDE_DIR}/sycl/ur_api.h      (does not exist → fatal error)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/includes")
file(WRITE "${CMAKE_BINARY_DIR}/includes/ur_api.h"
    "#pragma once\n#include \"${SYCL_INCLUDE_DIR}/ur_api.h\"\n")
set(CMAKE_CXX_FLAGS "/I\"${CMAKE_BINARY_DIR}/includes\" ${CMAKE_CXX_FLAGS}")
message(STATUS "ur_api.h shim → ${SYCL_INCLUDE_DIR}/ur_api.h")

find_package(Torch REQUIRED)

set(MODULE_NAME _ext)
Python_add_library(${MODULE_NAME} MODULE WITH_SOABI
    csrc/entry.cpp csrc/my_kernel_wrapper.cpp)

target_compile_definitions(${MODULE_NAME} PRIVATE TORCH_EXTENSION_NAME=${MODULE_NAME})

# Force oneAPI headers before conda's stale copies
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_include_directories(${MODULE_NAME} BEFORE PRIVATE
        "${SYCL_INCLUDE_DIR}/sycl" "${SYCL_INCLUDE_DIR}")
endif()

target_include_directories(${MODULE_NAME} PRIVATE
    "${CMAKE_BINARY_DIR}/includes" ${TORCH_INCLUDE_DIRS})

# Pre-built ESIMD DLL's import library (Step 1 must run first)
add_library(esimd_lib STATIC IMPORTED)
set_target_properties(esimd_lib PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lgrf_uni/my_kernel.lib")

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(${MODULE_NAME} PRIVATE
    ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} esimd_lib)

# scikit-build-core collects install() outputs into the wheel platlib
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION my_package)
install(FILES lgrf_uni/my_kernel.dll DESTINATION my_package)
```

---

## Step 4 — pyproject.toml

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "torch>=2.7.0", "wheel"]
build-backend = "scikit_build_core.build"

[project]
name = "my-kernels"
version = "0.0.1"
requires-python = ">=3.9"
dependencies = []

[tool.scikit-build]
cmake.build-type = "Release"
cmake.args = ["-GNinja"]
minimum-version = "build-system.requires"
wheel.py-api = "cp311"
wheel.license-files = []
wheel.packages = ["python/my_package"]   # pure-Python files to include

[tool.scikit-build.cmake.define]
CMAKE_CXX_COMPILER = "icx"
CMAKE_CXX_STANDARD = "20"
```

---

## Step 5 — Python package `__init__.py`

```python
import ctypes, os

_pkg_dir = os.path.dirname(os.path.abspath(__file__))

if os.name == "nt":
    os.add_dll_directory(_pkg_dir)          # tell Windows where to search for DLLs
    _dll = os.path.join(_pkg_dir, "my_kernel.dll")
    if os.path.isfile(_dll):
        ctypes.CDLL(_dll)                   # preload ESIMD DLL into the process
    else:
        raise FileNotFoundError(f"my_kernel.dll not found in {_pkg_dir}")

from my_package._ext import my_kernel      # noqa: E402, F401
from my_package.version import __version__ # noqa: F401
```

The `os.add_dll_directory` + `ctypes.CDLL` calls **must** happen before importing `_ext`.
On Windows, Python's import machinery doesn't search the package directory for side-by-side DLLs unless you tell it to. If you skip this, `_ext` will fail with a cryptic `ImportError: DLL load failed`.

---

## Step 6 — build.bat (all-in-one build script)

### Four known Windows bugs to fix in every build script

**Bug 1 — `setvars.bat` fails silently in `(x86)` paths**
When cmd.exe's current directory contains parentheses (e.g., after `pushd "C:\Program Files (x86)\Intel\oneAPI\advisor\latest\env"`), `call vars.bat` (without an explicit path) does NOT find `vars.bat` in the current directory. The result is `'vars.bat' is not recognized` for every component. setvars.bat still exits 0 in some contexts, but leaves icpx unconfigured.

Fix — call each component's `vars.bat` with its full absolute path:
```bat
if defined ONEAPI_ROOT (set "OA_ROOT=%ONEAPI_ROOT%") else (set "OA_ROOT=C:\Program Files (x86)\Intel\oneAPI")
for /f "delims=" %%d in ('dir /a:d /b "%OA_ROOT%"') do (
    if exist "%OA_ROOT%\%%d\latest\env\vars.bat" (
        call "%OA_ROOT%\%%d\latest\env\vars.bat"
    )
)
where icpx >nul 2>&1
if errorlevel 1 (echo ERROR: icpx not found after oneAPI init & exit /b 1)
```

**Bug 2 — `Scripts\cmake.exe` is a 108 KB Python launcher shim**
conda puts a tiny Python launcher at `Scripts\cmake.exe`. In redirected or non-interactive contexts it produces no output and exits with code 1. The real cmake binary is at `Lib\site-packages\cmake\data\bin\cmake.exe` (~12 MB).

Fix — use Python's `import cmake` to locate the real binary first:
```bat
"%PYEXE%" -c "import cmake,os; open('_tmp.txt','w').write(os.path.join(os.path.dirname(cmake.__file__),'data','bin','cmake.exe'))" 2>nul
if exist _tmp.txt (set /p CMAKE_EXE=<_tmp.txt & del _tmp.txt 2>nul)
if not defined CMAKE_EXE (
    for /f "tokens=*" %%c in ('where cmake 2^>nul') do if not defined CMAKE_EXE set "CMAKE_EXE=%%c"
)
```

**Bug 3 — `-w dist` in `python -m build` is wrong**
`-w` is short for `--wheel`, so `dist` is parsed as the positional `srcdir` argument — not the output directory. This causes `ERROR Source dist is not a directory`.

Fix — use `-o dist` (`--outdir`):
```bat
"%PYEXE%" -m build --wheel --no-isolation -o dist -C "build-dir=_cmake_build"
```

**Bug 4 — `ur_api.h` shim uses wrong subdirectory**
If a previous CMakeLists.txt wrote `${SYCL_INCLUDE_DIR}/sycl/ur_api.h`, that path does not exist. The file is at `${SYCL_INCLUDE_DIR}/ur_api.h` (one level up). See Step 3 above for the correct shim content.

### Complete `build.bat`

```bat
@echo off
setlocal
set "PROJ=%~dp0"
set "LOGFILE=%PROJ%build_log.txt"
set "ERRFILE=%PROJ%build_err.txt"
echo === Build start === > "%LOGFILE%"
echo %DATE% %TIME% >> "%LOGFILE%"

REM ── Python ───────────────────────────────────────────────────────────────
if defined CONDA_PREFIX (
    set "PYEXE=%CONDA_PREFIX%\python.exe"
) else (
    for /f "tokens=*" %%p in ('where python 2^>nul') do if not defined PYEXE set "PYEXE=%%p"
)
if not exist "%PYEXE%" (echo ERROR: Python not found. Activate conda env first. & exit /b 1)
echo PYEXE=%PYEXE% >> "%LOGFILE%"

REM ── cmake: prefer real binary (Bug 2 fix) ────────────────────────────────
"%PYEXE%" -c "import cmake,os; open('_tmp.txt','w').write(os.path.join(os.path.dirname(cmake.__file__),'data','bin','cmake.exe'))" 2>nul
if exist _tmp.txt (set /p CMAKE_EXE=<_tmp.txt & del _tmp.txt 2>nul)
if not defined CMAKE_EXE (
    for /f "tokens=*" %%c in ('where cmake 2^>nul') do if not defined CMAKE_EXE set "CMAKE_EXE=%%c"
)
if not defined CMAKE_EXE (echo ERROR: cmake not found. pip install cmake & exit /b 1)
echo CMAKE_EXE=%CMAKE_EXE% >> "%LOGFILE%"

REM ── ninja ────────────────────────────────────────────────────────────────
for /f "tokens=*" %%n in ('where ninja 2^>nul') do if not defined NINJA_EXE set "NINJA_EXE=%%n"
if not defined NINJA_EXE (echo ERROR: ninja not found. conda install ninja & exit /b 1)

REM ── MSVC ─────────────────────────────────────────────────────────────────
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo ERROR: vswhere not found. Install Visual Studio. & exit /b 1)
for /f "usebackq tokens=*" %%v in (`"%VSWHERE%" -latest -property installationPath`) do set "VS_INSTALL=%%v"
call "%VS_INSTALL%\VC\Auxiliary\Build\vcvarsall.bat" x64 >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo vcvarsall FAILED & exit /b 1)
echo vcvarsall OK >> "%LOGFILE%"

REM ── oneAPI (Bug 1 fix: full-path vars.bat calls) ──────────────────────────
if defined ONEAPI_ROOT (set "OA_ROOT=%ONEAPI_ROOT%") else (set "OA_ROOT=C:\Program Files (x86)\Intel\oneAPI")
for /f "delims=" %%d in ('dir /a:d /b "%OA_ROOT%"') do (
    if exist "%OA_ROOT%\%%d\latest\env\vars.bat" (
        call "%OA_ROOT%\%%d\latest\env\vars.bat"
    )
)
where icpx >nul 2>&1
if errorlevel 1 (echo ERROR: icpx not found after oneAPI init >> "%LOGFILE%" & exit /b 1)
echo oneAPI OK >> "%LOGFILE%"

REM ── torch root ───────────────────────────────────────────────────────────
"%PYEXE%" -c "import os,torch; open('_tmp.txt','w').write(os.path.dirname(torch.__file__))" 2>> "%ERRFILE%"
if errorlevel 1 (echo ERROR: torch not importable & exit /b 1)
set /p torch_root=<_tmp.txt & del _tmp.txt 2>nul
set "torch_root=%torch_root:\=/%"
echo torch_root=%torch_root% >> "%LOGFILE%"

REM ══ Step 1: ESIMD DLL ════════════════════════════════════════════════════
echo === Step 1: Build ESIMD DLL ===
cd /d "%PROJ%lgrf_uni"
if exist my_kernel.dll del /f my_kernel.dll
icpx kernels.cpp -shared -o my_kernel.dll ^
    -DBUILD_ESIMD_KERNEL_LIB ^
    -fsycl -fsycl-targets=spir64_gen ^
    -Xs "-device ptl-h -options -doubleGRF" ^
    -O3 >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo DLL BUILD FAILED >> "%LOGFILE%" & exit /b 1)
echo DLL build OK >> "%LOGFILE%" & echo DLL build OK

REM ══ Step 2: cmake build (.pyd) ════════════════════════════════════════════
echo === Step 2: cmake build ===
cd /d "%PROJ%"
if exist _cmake_build rmdir /s /q _cmake_build
"%CMAKE_EXE%" -GNinja "-DCMAKE_MAKE_PROGRAM=%NINJA_EXE%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER=icx ^
    -DCMAKE_CXX_STANDARD=20 ^
    "-DCMAKE_PREFIX_PATH=%torch_root%" ^
    "-DPython_EXECUTABLE=%PYEXE%" ^
    -B _cmake_build -S . >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo CMAKE CONFIGURE FAILED >> "%LOGFILE%" & exit /b 1)
"%CMAKE_EXE%" --build _cmake_build --config Release -j >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo CMAKE BUILD FAILED >> "%LOGFILE%" & exit /b 1)
echo cmake build OK >> "%LOGFILE%" & echo cmake build OK

REM ══ Step 3: Copy artifacts ════════════════════════════════════════════════
echo === Step 3: Copy artifacts ===
for /f "tokens=*" %%f in ('dir /b "_cmake_build\_ext*.pyd" 2^>nul') do (
    copy /y "_cmake_build\%%f" "python\my_package\" >> "%LOGFILE%"
)
copy /y "lgrf_uni\my_kernel.dll" "python\my_package\" >> "%LOGFILE%"
echo Artifacts copied

REM ══ Step 4: Wheel ════════════════════════════════════════════════════════
echo === Step 4: Build wheel ===
if exist dist rmdir /s /q dist
set "CMAKE_PREFIX_PATH=%torch_root%"
"%PYEXE%" -m build --wheel --no-isolation -o dist ^
    -C "build-dir=_cmake_build" >> "%LOGFILE%" 2>> "%ERRFILE%"
if errorlevel 1 (echo WHEEL BUILD FAILED >> "%LOGFILE%" & exit /b 1)
for /f "tokens=*" %%w in ('dir /b "dist\*.whl" 2^>nul') do echo Wheel: dist\%%w
echo %DATE% %TIME% >> "%LOGFILE%"
echo ===FINISHED=== >> "%LOGFILE%"
endlocal & exit /b 0
```

---

## Invoking from Claude Code (bash shell on Windows)

Git Bash / MSYS2 communicates with cmd.exe via pipes. The pipe makes cmd.exe non-interactive, which triggers Bug 1. Use a separate `run_build.bat` that sets up the conda env by directly manipulating environment variables (bypassing `conda activate`):

### `run_build.bat`
```bat
@echo off
REM Set conda env directly — activate.bat blocks in piped cmd.exe sessions
set "CONDA_PREFIX=C:\Users\Local_Admin\miniforge3\envs\my_env"
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%PATH%"
REM Pre-set cmake to the real binary so build.bat doesn't have to detect it
set "CMAKE_EXE=%CONDA_PREFIX%\Lib\site-packages\cmake\data\bin\cmake.exe"
cd /d D:\path\to\my_kernel
call "D:\path\to\my_kernel\build.bat"
```

Invoke from bash:
```bash
echo 'call D:\path\to\run_build.bat > D:\path\to\build_out.log 2>&1' | cmd.exe
```

After it finishes, check success and the wheel:
```bash
cat /d/path/to/build_log.txt | tail -5    # should end with ===FINISHED===
ls /d/path/to/dist/*.whl                  # e.g. my_kernels-0.0.1-cp311-abi3-win_amd64.whl
```

---

## Environment setup (first time)

```cmd
conda create -n my_env python=3.11
conda activate my_env
conda install -c conda-forge ninja
pip install cmake scikit-build-core wheel
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Required system software:
- **Intel oneAPI Base Toolkit 2025.x** — provides `icpx`, `icx`, SYCL headers
- **Visual Studio 2022** with "Desktop development with C++" workload
- **miniforge or miniconda** — for conda env management

---

## Install and use the wheel

```cmd
pip install dist\my_kernels-0.0.1-cp311-abi3-win_amd64.whl --force-reinstall --no-deps
```

```python
import my_package

# tensors must be on XPU and contiguous
result = my_package.my_kernel(input_tensor)
```

---

## Debugging checklist

| Symptom | Cause | Fix |
|---------|-------|-----|
| `'vars.bat' is not recognized` | Bug 1: cmd.exe in `(x86)` path | Full-path vars.bat loop (see Bug 1) |
| cmake exits silently, rc=1 | Bug 2: `Scripts\cmake.exe` shim | Use `import cmake` to find real cmake |
| `ERROR Source dist is not a directory` | Bug 3: `-w dist` parsed as srcdir | Change to `-o dist` |
| `fatal error: '.../sycl/ur_api.h' file not found` | Bug 4: wrong shim path | Use `${SYCL_INCLUDE_DIR}/ur_api.h` |
| `ImportError: DLL load failed` | `__init__.py` missing DLL preload | Add `os.add_dll_directory` + `ctypes.CDLL` before `_ext` import |
| `icpx: command not found` | oneAPI not initialized | Run oneAPI vars.bat or check `OA_ROOT` |
| `torch not importable` | Wrong conda env | Check `CONDA_PREFIX`, activate correct env |
| cmake warns `IntelSYCL not found` | `CMAKE_CXX_COMPILER=icx` not in PATH | Ensure oneAPI initialized before cmake |
