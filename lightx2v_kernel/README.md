# LightX2V Kernels

Kernel projects are organized by hardware backend. Build and install each project
from its own directory:

| Backend | Source directory | Python package | Build instructions |
| --- | --- | --- | --- |
| NVIDIA CUDA | `nvidia/` | `lightx2v_kernel_nvidia` | [NVIDIA README](nvidia/README.md) |
| Intel XPU | `intel_xpu/` | `sycl_kernels` | [Intel XPU README](intel_xpu/README.md) |

LightX2V prefers `lightx2v_kernel_nvidia` and falls back to the previously built
`lightx2v_kernel` package when the new package is not installed.
