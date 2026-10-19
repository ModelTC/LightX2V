---
name: sycl-esimd-to-python-wheel
description: >
  Package SYCL/ESIMD kernels as Python-importable wheels or extend LightX2V's
  existing XPU extension. Covers the shared-library bridge, PyTorch XPU stream,
  CMake and wheel registration, Windows build troubleshooting and the current
  Linux/Windows repository build paths.

---

# SYCL/ESIMD Kernel → Python Wheel

## Current repository entry points

The maintained LightX2V project is [lightx2v_kernel/intel_xpu](../../../../../../lightx2v_kernel/intel_xpu/README.md) at the repository root. The [older Windows reference package](assets/lightx2v_kernel_xpu/README.md) is retained with its original sources and prebuilt artifacts for standalone examples and historical comparison. It is not the current runtime implementation; check its documented environment and ABI before reusing binaries.

Before copying the legacy bridge, read its [validation and device-cache limitations](assets/lightx2v_kernel_xpu/README.md#reference-package-limitations). Its success messages and exit status do not establish numerical correctness. An adapted validation entry must propagate failed checks; wrappers must enforce device/shape contracts and keep caches valid for the requested device and stream lifecycle.

| Source | Responsibility |
|---|---|
| [CMakeLists.txt](../../../../../../lightx2v_kernel/intel_xpu/CMakeLists.txt) | Target registration, compiler-matched UR shim, SYCL sources, library linkage and wheel installs |
| [build.sh](../../../../../../lightx2v_kernel/intel_xpu/build.sh) / [build.bat](../../../../../../lightx2v_kernel/intel_xpu/build.bat) | Existing Linux/Windows build and smoke-check orchestration |
| [entry.cpp](../../../../../../lightx2v_kernel/intel_xpu/csrc/entry.cpp) | Legacy pybind exports: SDP, INT8/FP8/INT4 GEMM and FP8 cache helpers |
| [Python package](../../../../../../lightx2v_kernel/intel_xpu/python/sycl_kernels/__init__.py) | Library loading, wrappers, CUTE/SLA and norm/RoPE capability probes |
| [pyproject.toml](../../../../../../lightx2v_kernel/intel_xpu/pyproject.toml) / [build_version.py](../../../../../../lightx2v_kernel/intel_xpu/build_version.py) | Wheel metadata and target-specific version suffixes |
| [test/](../../../../../../lightx2v_kernel/intel_xpu/test/) | Operator correctness and benchmark entry points |

For a LightX2V change, adapt the existing project. For a standalone wheel, use the bridge layout below and adapt CMake targets, package names, exports and install destinations to the requested output project; do not require changes to LightX2V itself. Read [Windows build notes](references/windows-build-notes.md) for non-interactive shell and tool-discovery failures. Reviewing this documentation alone does not require running a build or inference.

## Pipeline overview

```
lgrf_uni/kernels.cpp   ──icpx──►  my_kernel.dll + my_kernel.lib
                                         │
csrc/entry.cpp                           │ (linked via .lib)
csrc/wrapper.cpp       ──cmake──►  _ext.cp311-win_amd64.pyd
                                         │
python/my_package/                       │ (copied in)
  __init__.py          ──build──►  dist/<wheel produced by the build>.whl
  _ext.*.pyd
  my_kernel.dll
```

For the legacy SDP bridge, the dependency order is:
1. **ESIMD DLL** — compiled with `icpx -fsycl`, AOT for the target GPU
2. **Python extension** — links the DLL's `.lib`; oneDNN sources also require SYCL compilation through `add_sycl_to_target`. Linux uses `icpx` and shared `.so` libraries.

The current CMake project additionally builds norm, RoPE and optional CUTE libraries loaded through `torch.ops`; not every new kernel belongs in the legacy DLL/pybind bridge.

---

## Illustrative legacy bridge layout

```
my_kernel/
├── CMakeLists.txt
├── pyproject.toml
├── build.bat                     ← all-in-one build script
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

This command illustrates the PTL-H legacy DLL. On Linux, `XPU_TARGET=bmg|ptl-h` selects the CMake/CUTE target, but `lgrf_uni/build.sh` independently hard-codes `ptl-h` for the legacy SDP library. Inspect both stages when retargeting; preserve `doubleGRF` when required by the selected kernel tile.

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

## Step 3 — Register the extension and install artifacts

Update the existing CMake target with the new wrapper and its SYCL source, then add the export in `csrc/entry.cpp` and the Python wrapper/import. For a separate `torch.ops` library, follow the existing RMSNorm/RoPE targets and loader instead.

- Use `c10::xpu::getCurrentXPUStream(device.index()).queue()` so work follows PyTorch's selected stream; validate device, dtype, shape and layout at the wrapper boundary.
- Keep `TORCH_EXTENSION_NAME`, include order, `add_sycl_to_target`, link libraries and `install()` destinations consistent with the current target.
- The current CMake-generated UR shim includes `${SYCL_INCLUDE_DIR}/ur_api.h`; do not copy the historical snapshot's machine-specific header or insert an extra `/sycl/` path segment.
- The kernel template above waits for completion. If removing the wait for asynchronous integration, establish input/output/temporary lifetimes and stream ordering first; validate non-default streams.

## Step 4 — Wheel metadata

Reuse `pyproject.toml` and `build_version.py`. Linux versions use `+bmg` / `+ptlh`; Windows returns the base version. The filename/tag is produced by the active build configuration, so inspect `dist/` instead of hard-coding a wheel name. A Python wheel tag does not prove compatibility with a different PyTorch/oneDNN runtime or GPU target.

## Step 5 — Python library loading

Reuse `python/sycl_kernels/__init__.py`. On Windows it adds DLL search paths and preloads the ESIMD DLL before `_ext`; on Linux it preloads the ESIMD `.so` with global visibility. Separate norm/RoPE/CUTE libraries use `torch.ops.load_library()`.

Add an explicit export or wrapper for the new op and expose a capability probe only if callers need one. Do not treat successful package import or `hasattr()` as proof that an optional op works: some exports are error-raising stubs when a library cannot load. Exercise the real function on a small supported input.

---

## Step 6 — Build, install and validate

Use the existing build scripts in an environment with matching oneAPI, oneDNN and PyTorch XPU. The repository README records an example environment, not a requirement to install a particular future version from this skill.

Linux, from the repository root:

```bash
source /opt/intel/oneapi/setvars.sh
cd lightx2v_kernel/intel_xpu
XPU_TARGET=bmg ./build.sh
ls dist/*.whl
```

`build.sh` supports `PYTHON`, `CMAKE`, `CXX`, `BUILD_JOBS`, `CUTLASS_SYCL_ROOT` and `SKIP_TESTS`. It enables CUTE FMHA; without `CUTLASS_SYCL_ROOT`, CMake fetches a pinned sycl-tla revision. Confirm that dependency/network access is part of the build task. `SKIP_TESTS=1` skips the script's smoke check only, not compilation or runtime prerequisites, and does not validate the wheel.

The command above selects BMG for CMake/CUTE, while `lgrf_uni/build.sh` still builds legacy SDP for PTL-H. Before running legacy SDP on BMG, inspect and adapt that script's AOT target for the device/compiler and validate the result. A `+bmg` wheel suffix or passing CUTE smoke check does not establish legacy SDP compatibility.

Windows, in cmd.exe with the intended environment active:

```cmd
cd /d C:\path\to\LightX2V\lightx2v_kernel\intel_xpu
call build.bat
dir dist\*.whl
```

Windows `build.bat` currently hard-codes `-device ptl-h` for the legacy ESIMD DLL and disables CUTE. It does not read `XPU_TARGET` or pass it to CMake. Setting that variable alone does not retarget the Windows build; inspect both build stages before using another device. CMake rejects CUTE on Windows because of the recorded correctness issue.

Both scripts create build/package artifacts inside their project directory; Windows also replaces `_cmake_build` and `dist`. If the user requires all temporary/build files elsewhere, copy the project to that temporary directory and build the copy. Do not execute the repository build scripts during documentation maintenance.

Install exactly the wheel from this build:

```bash
python -m pip install /path/to/selected.whl --force-reinstall --no-deps
```

The scripts use `python -m pip wheel . --no-build-isolation`; do not replace it with the old copied `python -m build` launcher. Inspect the wheel's library contents and imported package path, then select tests for the changed operator:

```bash
# From the built project's directory, with each kernel's target matching the device:
python test/test_sdp.py
python -m pytest test/test_int8.py
python test/test_fp8.py
```

Run only relevant tests; CUTE, SLA, RMSNorm and RoPE have separate tests under `test/`. Compare numerical results with the existing reference, validate a non-default stream and repeated calls, and report skipped hardware checks separately. Import success is not numerical or performance acceptance.

---

## Debugging checklist

| Symptom | Check |
|---|---|
| `icpx` / `icx` missing | Active oneAPI environment and compiler path |
| CMake exits without useful output | Build logs and actual CMake executable; Windows script already locates the pip package binary when available |
| `ur_api.h` mismatch | Compiler-matched generated include path and header order in CMake |
| `DLL load failed` / unresolved oneDNN symbols | Packaged DLL/SO, loader search path, architecture and matching headers/runtime |
| Package imports but op fails | Actual exported function, optional-library probe and small tensor call; check for error stubs |
| Unsupported kernel/device | AOT target in both shared library and extension, not only wheel version text |
| CUTE unavailable on Windows | Current CMake deliberately disables it; choose an adapter with a verified compatible contract |
