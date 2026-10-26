# lightx2v_kernel_xpu

> Historical Windows reference package. This directory preserves the original small
> DLL → PyTorch extension → wheel example and its bundled binaries; it is not a
> byte-for-byte copy of the maintained repository-root project. For current
> LightX2V operators and Linux/CUTE support, use the
> [maintained project](../../../../../../../../lightx2v_kernel/intel_xpu/README.md) and
> [integration guide](../../SKILL.md).
> The environment and measurements below belong to this older snapshot. Bundled
> DLL/PYD/LIB/EXP files require matching runtime/ABI and GPU checks before reuse.
> `includes/ur_api.h` retains a machine-specific path. CMake leaves it unchanged
> and generates a separate shim under the build directory using
> `${SYCL_INCLUDE_DIR}/ur_api.h`; check that generated path against the compiler.
> Copy the example to the task's temporary/output directory when experimenting.

SYCL/ESIMD custom kernels for LightX2V inference on Intel Arc GPU (Xe3 / PTL-H).

Exposed as the Python package `sycl_kernels`:

| Function | Description |
|----------|-------------|
| `sdp(Q, K, V)` | ESIMD Flash Attention — `[B, L, H, 128]` fp16/bf16, PTL-H doubleGRF |
| `onednn_w8a16_fp8(x, qweight, scales[, bias])` | W8A16 GEMM — fp16/bf16 activations × FP8_E4M3 weights, per-column scale |
| `onednn_w4a16(x, weight, scales, zeros[, bias])` | W4A16 GEMM — fp16/bf16 activations × INT4 packed weights |

Historical test environment: **Intel Arc B390 GPU** (PTL-H / Xe3), PyTorch 2.9.1+xpu, oneAPI 2025.2. The architecture follows [Intel's device table](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html); the measurements below have not been revalidated by this documentation update.

## Reference package limitations

- `test/test_sdp.py` prints correctness failures but ignores the returned booleans. `build.bat` can therefore print `TEST PASSED` after a numerical failure, continues after a test failure, and returns zero even through its failure label. Neither its final exit code nor its success message is an acceptance result. Before using a copy for automated validation, aggregate checks, return nonzero on failure, and stop the build's validation path when they fail.
- `csrc/sdp.cpp` caches `normAlpha` by head count only. The snapshot is unsuitable for changing XPU devices in one process: the cache can retain a tensor from the previous device. It also lacks Q/K/V same-device and mutual K/V shape checks. Before reusing this wrapper, validate those contracts and remove the cache or key it by device, then test the intended device/stream lifecycle.
- `test/test_linear.py` is not included. W4A16 has no bundled dedicated Python test here; add a check against a dequantized reference before treating it as validated.

The sources and binaries remain a historical snapshot. Use the [current integration guide](../../SKILL.md) to adapt the maintained project; the limits above do not establish the validation status of that separate project.

---

## Requirements

| Component | Version |
|-----------|---------|
| Intel oneAPI Base Toolkit | 2025.2 (provides `icpx`, `icx`, SYCL/ESIMD headers) |
| oneDNN | 2025.2.0 (must match oneAPI version) |
| Visual Studio | 2022, Desktop development with C++ workload |
| PyTorch | 2.9.1+xpu |
| Python | 3.11 |
| miniforge / miniconda | latest |

> The oneDNN, oneAPI, and PyTorch versions must align. Mixing versions causes linker or runtime errors.

---

## Environment Setup

```cmd
conda create -n lightx2v_kernel python=3.11
conda activate lightx2v_kernel
conda install -c conda-forge ninja
pip install cmake scikit-build-core wheel
pip install onednn==2025.2.0 onednn_devel==2025.2.0
pip install --no-cache-dir torch==2.9.1+xpu torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/xpu
```

---

## Build

Open **cmd.exe** (not PowerShell), activate the conda env, then run:

```cmd
conda activate lightx2v_kernel
cd D:\path\to\lightx2v_kernel_xpu
call build.bat
```

`build.bat` auto-detects Visual Studio (via vswhere), oneAPI, cmake, and the active Python — no hard-coded paths. It runs five steps:

| Step | Action | Output |
|------|--------|--------|
| 1 | Build ESIMD DLL with `icpx` (PTL-H, doubleGRF) | `lgrf_uni\esimd.unify.lgrf.dll` |
| 2 | Build PyTorch C++ extension with CMake + ninja | `_cmake_build\_ext.cp311-win_amd64.pyd` |
| 3 | Copy `.pyd` + `.dll` to package dir | `python\sycl_kernels\` |
| 4 | Run smoke test | `test\test_sdp.py` |
| 5 | Build wheel (`pip wheel --no-build-isolation`) | `dist\sycl_kernels-0.0.1-cp311-win_amd64.whl` |

Logs: `build.log` (full output), `build.err` (compiler warnings).

The historical launcher can return exit code 0 after failure. Check `build.log`
for `===FAILED===` / `===FINISHED===`, inspect test results and confirm that the
expected wheel was freshly produced; the process exit code alone is insufficient.

> **Note on wheel build**: Step 5 uses `pip wheel --no-build-isolation` rather than
> `python -m build --no-isolation`. The latter fails when `cmake` is only a system binary
> (not pip-installed): scikit-build-core dynamically injects `cmake>=3.22` as a build
> requirement and the `build` package rejects it. `pip wheel` skips that pre-check and
> locates cmake directly from PATH.

---

## Install Wheel

```cmd
REM Replace the placeholder with the wheel printed by this build.
python -m pip install "dist\<actual-wheel-filename>.whl" --force-reinstall --no-deps
```

Keep the complete Python, ABI and platform tags from the generated filename; do not reconstruct the filename from the package version alone.

---

## Usage

```python
import torch
import sycl_kernels

# ── Flash Attention (SDP) ─────────────────────────────────────────────────────
# Q, K, V : [B, L, H, 128]  fp16 or bf16  on XPU  (B=1, D=128 required)
# Returns : [B, L, H, 128]  same dtype as input
# Dispatch: fp16 → sdp_fp16 kernel,  bf16 → sdp_bf16io kernel
Q = torch.randn(1, 14040, 12, 128, dtype=torch.bfloat16, device="xpu")
K = torch.randn(1, 14040, 12, 128, dtype=torch.bfloat16, device="xpu")
V = torch.randn(1, 14040, 12, 128, dtype=torch.bfloat16, device="xpu")
out = sycl_kernels.sdp(Q, K, V)                    # [1, 14040, 12, 128] bf16

# ── W8A16 FP8 GEMM ────────────────────────────────────────────────────────────
# x       : [M, K]  fp16 or bf16     on XPU
# qweight : [N, K]  float8_e4m3fn    on XPU
# scales  : [N, 1]  fp32             on XPU  (per-output-channel absmax scale)
# bias    : [N]     fp16/bf16        on XPU  (optional)
# Returns : [M, N]  same dtype as x
out = sycl_kernels.onednn_w8a16_fp8(x, qweight, scales)
out = sycl_kernels.onednn_w8a16_fp8(x, qweight, scales, bias)

# ── W4A16 GEMM ────────────────────────────────────────────────────────────────
# x      : [M, K]    fp16 or bf16    on XPU
# weight : [N, K//2] int8 (INT4×2)   on XPU  (two INT4 packed per byte)
# scales : [N, G]    fp16 or bf16    on XPU  (G = K / group_size)
# zeros  : [N, G]    fp16 or bf16    on XPU
# bias   : [N]       fp16/bf16       on XPU  (optional)
# Returns: [M, N]    same dtype as x
out = sycl_kernels.onednn_w4a16(x, weight, scales, zeros)
out = sycl_kernels.onednn_w4a16(x, weight, scales, zeros, bias)
```

---

## Tests

```cmd
REM Flash Attention — correctness + performance (both fp16 and bf16):
python test\test_sdp.py

REM W8A16 FP8 GEMM — correctness + benchmark:
python test\test_fp8.py

```

These are historical diagnostic runs; see [Reference package limitations](#reference-package-limitations) before using their output or exit status for acceptance. A W4A16 test is not bundled.

> Running from source without installing the wheel: set PYTHONPATH first.
> ```cmd
> set PYTHONPATH=D:\path\to\lightx2v_kernel_xpu\python;%PYTHONPATH%
> python test\test_sdp.py
> ```

---

## Performance (Intel Arc B390, PTL-H / Xe2)

### Flash Attention — LightX2V WAN inference shapes

| Shape | Kernel | ms/iter | TFLOPS | vs torch SDPA |
|-------|--------|--------:|-------:|:-------------:|
| Self-attn Q/K/V = [14040, 12, 128] | sdp bf16 | 33.9 | 35.7 | **1.47×** |
| Self-attn Q/K/V = [14040, 12, 128] | sdp fp16 | 35.4 | 34.2 | 1.12× |
| Self-attn Q/K/V = [14040, 12, 128] | torch SDPA bf16 | 49.9 | 24.2 | baseline |
| Cross-attn Q=[14040,12,128] K/V=[512,12,128] | sdp bf16 | 1.58 | 27.9 | 1.07× |
| Cross-attn Q=[14040,12,128] K/V=[512,12,128] | sdp fp16 | 1.81 | 24.5 | — |

### Flash Attention kernel architecture

`sdp()` dispatches to one of two ESIMD kernels depending on input dtype:

| Kernel | I/O dtype | QK DPAS acc | SV DPAS acc | Notes |
|--------|-----------|-------------|-------------|-------|
| `sdp_fp16` | fp16 | fp32 | **fp16** | Native fp16 accumulator on Xe2 — no conversion overhead |
| `sdp_bf16io` | bf16 | bf16 | **fp16** | bf16 I/O, fp16 internal; V bf16→fp16 hidden in DPAS |

Both kernels use:
- doubleGRF (256 registers / thread) compiled AOT for PTL-H
- SLM-based ping-pong V buffering
- Barrier moved before compensation (SLM loads overlap with compensation ALU)
- Interleaved l=0/l=1 compensation overlaps with SxV DPAS pipeline
- Precomputed `normAlpha` cached across calls (avoids per-call XPU fill kernel)
- Explicit `.wait()` on each submit — prevents output tensor use-after-free when caller discards the return value

### W8A16 FP8 GEMM

Run `python test\test_fp8.py` for benchmark numbers on your hardware.
FP16 activations use the optimised `jit:gemm` path; BF16 falls back to `ocl:ref` on PTL-H.

---

## Project Layout

```
lightx2v_kernel_xpu\
├── lgrf_uni\                   # ESIMD DLL source (compiled by icpx)
│   ├── sdp_kernels.cpp         # DLL entry points: sdp_fp16, sdp_bf16io
│   ├── single_kernels\
│   │   ├── flash.attn.b.mha128.fp16.opt.h   # FP16 Flash Attention kernel
│   │   └── flash.attn.b.mha128.bf16io.h     # BF16 I/O Flash Attention kernel
│   ├── esimd_kernel_api.h      # DLL export macro
│   └── build.bat               # icpx compile command (called by build.bat)
├── csrc\                       # PyTorch C++ extension source (compiled by icx via CMake)
│   ├── entry.cpp               # pybind11 module registration
│   ├── sdp.cpp                 # sdp() Python wrapper — dtype dispatch + normAlpha cache
│   ├── sdp_kernels.h           # DLL function declarations
│   ├── onednn.cpp              # W4A16 GEMM via oneDNN
│   ├── onednn_fp8.cpp          # W8A16 FP8 GEMM via oneDNN
│   └── utils.h                 # get_queue() helper
├── python\sycl_kernels\        # Python package (importable after Step 3)
│   ├── __init__.py             # DLL load + re-export
│   └── version.py
├── test\
│   ├── test_sdp.py             # Flash Attention correctness + perf
│   └── test_fp8.py             # W8A16 FP8 correctness + perf
├── CMakeLists.txt              # CMake build for .pyd
├── pyproject.toml              # scikit-build-core wheel config
└── build.bat                # Full build script: DLL + pyd + test + wheel
```
