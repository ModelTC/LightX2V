# NVIDIA GPU

English | [简体中文](cuda.md)

## Default behavior and platform isolation

The CUDA path selects `TorchBlockOffload` through `_DEFAULT_BLOCK_OFFLOAD_BACKENDS` in [offload.py](../../../../lightx2v_platform/base/offload.py), using the registered platform name `cuda`. It requires no changes to [CudaDevice](../../../../lightx2v_platform/base/nvidia.py). A platform class's explicit `block_offload_backend` takes precedence over the default mapping. This backend allocates pinned CPU storage and device byte buffers and submits copies on the current stream. Reuse the common layout and manager rather than creating a separate CUDA model loader.

Before changing code, record the target model's configuration and key outputs with contiguous layout disabled. Preserve CUDA's default operators, checkpoint conversions, precision, shapes, and scheduling. After common code changes, validate both the per-tensor baseline and contiguous paths.

- NVIDIA execution must not depend on `torch_npu`, CANN, or successful registration of Ascend operators. Load platform modules through the existing platform registration entry points.
- Keep dependencies needed only by a selected operator in that operator's loading branch. See [QwenImageTransformerInfer](../../../../lightx2v/models/networks/qwen_image/infer/transformer_infer.py): Qwen Triton modulation kernels are imported only for `modulate_type=triton`, preserving the default selection.
- Do not replace CUDA's default attention, RoPE, or norm with generic PyTorch implementations merely to support NPU.
- Reporting `device.type == "cuda"` or inheriting `CudaDevice` does not enable this capability for another vendor. The default mapping matches only the registered platform name `cuda`; other platforms must declare support. Preserve the original device definition in `nvidia.py` without adding offload backend attributes.

## Configuration and weights

Select the intended NVIDIA device in the entry point. Set `PLATFORM=cuda` explicitly when needed to avoid inheriting an NPU platform from the shell. CUDA examples use `CUDA_VISIBLE_DEVICES`; do not combine NPU environment selection into the same launcher.

Hold the main dtype, sensitive-layer dtype, and quantization scheme constant when comparing variants. The current Wan CUDA example uses FP8-vLLM weights with BF16 computation, while the Qwen example uses original BF16 weights. These are [model examples](model-examples_en.md), not restrictions requiring all CUDA weights to use FP8 or BF16.

Before adding a quantization scheme, check that the operator's `describe_storage()` matches its loading behavior, especially transposition, scale/bias precision, and auxiliary device state. An existing quantized compute kernel does not establish a contiguous-layout storage contract.

Use the same platform operators for baseline and contiguous runs. Record the weights actually transferred and their byte counts; total model parameter size is not the H2D byte count per step.

## Validation priorities

Select relevant checks from [Validation](validation_en.md), paying particular attention to:

1. CUDA import isolation: common loading and model weight classes must initialize even when imports of Ascend operator modules are blocked.
2. Stable addresses for both slots and pinned CPU sources. Synchronize as needed before checking values and strides after asynchronous copies.
3. Correct switching between multiple registered groups and completion of every transfer during cleanup.
4. Baseline/contiguous generation comparisons with matching configurations, plus regression checks for CUDA's default computation before and after the change.

Copying small matrices does not validate complete attention, RoPE, text encoding, or VAE behavior. State the scope of smoke tests with reduced steps or resolution. NPU changes that touch shared functions still require the applicable CUDA checks above.

When changing stream or event protocols, validate dependencies and overlap. Do not hide errors by removing synchronization or adding whole-device synchronization. Measure performance only when requested.
