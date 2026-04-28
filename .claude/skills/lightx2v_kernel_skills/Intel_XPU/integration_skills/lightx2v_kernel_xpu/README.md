# lightx2v_kernel_xpu

SYCL/ESIMD custom kernels for LightX2V inference on Intel Arc GPU (Xe2 / PTL-H).

Exposed as the Python package `sycl_kernels`:

| Function | Description |
|----------|-------------|
| `sdp(Q, K, V)` | ESIMD Flash Attention — `[B, L, H, 128]` fp16/bf16, PTL-H doubleGRF |
| `onednn_w8a16_fp8(x, qweight, scales[, bias])` | W8A16 GEMM — fp16/bf16 activations × FP8_E4M3 weights, per-column scale |
| `onednn_w4a16(x, weight, scales, zeros[, bias])` | W4A16 GEMM — fp16/bf16 activations × INT4 packed weights |

Tested on **Intel Arc B390 GPU** (PTL-H / Xe2), PyTorch 2.9.1+xpu, oneAPI 2025.2.

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
call build_la.bat
```

`build_la.bat` auto-detects Visual Studio (via vswhere), oneAPI, cmake, and the active Python — no hard-coded paths. It runs five steps:

| Step | Action | Output |
|------|--------|--------|
| 1 | Build ESIMD DLL with `icpx` (PTL-H, doubleGRF) | `lgrf_uni\esimd.unify.lgrf.dll` |
| 2 | Build PyTorch C++ extension with CMake + ninja | `_cmake_build\_ext.cp311-win_amd64.pyd` |
| 3 | Copy `.pyd` + `.dll` to package dir | `python\sycl_kernels\` |
| 4 | Run smoke test | `test\test_sdp.py` |
| 5 | Build wheel (`pip wheel --no-build-isolation`) | `dist\sycl_kernels-0.0.1-cp311-win_amd64.whl` |

Logs: `build_la_log.txt` (full output), `build_la_err.txt` (compiler warnings).

> **Note on wheel build**: Step 5 uses `pip wheel --no-build-isolation` rather than
> `python -m build --no-isolation`. The latter fails when `cmake` is only a system binary
> (not pip-installed): scikit-build-core dynamically injects `cmake>=3.22` as a build
> requirement and the `build` package rejects it. `pip wheel` skips that pre-check and
> locates cmake directly from PATH.

---

## Install Wheel

```cmd
pip install dist\sycl_kernels-0.0.1-cp311-win_amd64.whl --force-reinstall --no-deps
```

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

REM W4A16 GEMM — correctness + benchmark:
python test\test_linear.py
```

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
│   └── build.bat               # icpx compile command (called by build_la.bat)
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
│   ├── test_fp8.py             # W8A16 FP8 correctness + perf
│   └── test_linear.py         # W4A16 correctness + perf
├── CMakeLists.txt              # CMake build for .pyd
├── pyproject.toml              # scikit-build-core wheel config
└── build_la.bat                # Full build script: DLL + pyd + test + wheel
```
