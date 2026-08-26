# Intel XPU Multi-Device Usage Notes

LightX2V uses `torchrun` to launch multi-device inference on Intel XPU and uses
PyTorch XCCL/oneCCL for inter-device communication. Start with a script under
`dist_infer/` that matches the target model and parallel strategy, then adjust
it as needed.

## 1. Pre-launch checks

- Verify that PyTorch recognizes every device:

  ```bash
  python -c 'import torch; print(torch.xpu.is_available(), torch.xpu.device_count())'
  ```

- PyTorch, the Intel GPU driver, Level Zero, and oneCCL must use mutually
  compatible versions. If Intel provides an environment setup script, source
  it before launching LightX2V.
- All processes must use the same Python environment, model files, and
  LightX2V source tree. Make sure `PLATFORM=intel_xpu` is set before launch.

## 2. Match the device count to the parallel configuration

The parallel sizes in the configuration file must satisfy:

```text
number of torchrun processes = tensor_p_size × seq_p_size × cfg_p_size
```

Any parallel size that is omitted defaults to `1`. For example:

```json
"parallel": {
  "tensor_p_size": 2,
  "seq_p_size": 2,
  "cfg_p_size": 1
}
```

Launch this configuration with four processes:

```bash
torchrun --standalone --nproc_per_node=4 -m lightx2v.infer ...
```

LightX2V reports an error if the parallel sizes do not match the process count.
Each process normally corresponds to one visible XPU device.

## 3. Select visible devices correctly

Use `ZE_AFFINITY_MASK` to select Intel XPU devices. Do not substitute
`CUDA_VISIBLE_DEVICES`:

```bash
export ZE_AFFINITY_MASK=0,1,2,3
```

`--nproc_per_node` must not exceed the number of visible XPU devices. On
devices with tiles, Level Zero may identify a subdevice as `device.tile`, such
as `0.0`. Check the device numbering reported by `xpu-smi` or `sycl-ls` on the
current machine so that tiles are not mistaken for physical devices. When
running multiple jobs on one host, assign non-overlapping devices and a
different rendezvous port to each job.

## 4. oneCCL communication settings

Ulysses sequence parallelism uses all-to-all communication. Keep the following
oneCCL environment variables from the repository's example scripts unless the
current driver and oneCCL combination has been validated with other settings:

```bash
export CCL_SYCL_ALLTOALL_ARC_LL=1
export CCL_SYCL_ALLTOALL_TMP_BUF=1
export CCL_SYCL_CCL_BARRIER=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
```

These variables are already set in the SP and SP+TP examples under
`dist_infer/`. Pure TP scripts do not need the all-to-all-specific settings.
Do not apply NVIDIA NCCL tuning variables directly to XPU; the XPU backend is
`xccl`.

## 5. Choosing a parallel strategy

- **TP (tensor parallelism)**: Use it when the model does not fit on one device
  or when linear-layer computation needs to be partitioned. Relevant
  dimensions, such as the number of attention heads, must be divisible by
  `tensor_p_size`.
- **SP (sequence parallelism)**: Use it for long videos or high resolutions.
  With Ulysses, the number of attention heads and sequence-related dimensions
  must satisfy the partitioning requirements.
- **CFG parallelism**: Two device groups compute the conditional (positive
  prompt) and unconditional or negative-condition (negative prompt) branches
  separately, then combine their predictions. It is suitable for models that
  actually perform conventional Classifier-Free Guidance during inference and
  have similarly expensive branches. The usual settings are `enable_cfg=true`
  and `cfg_p_size=2`. Compared with serial CFG, it reduces the latency of the
  two DiT forwards in each denoising step, but requires at least two devices and
  does not reduce the memory required by either individual branch. Use TP or SP
  first if the goal is to resolve an out-of-memory error in a single branch.
- **SP + TP**: The total device count is the product of the two parallel sizes,
  and communication overhead is higher. Validate TP and SP separately before
  combining them to make configuration and communication issues easier to
  isolate.

Multi-device inference does not always scale linearly. For small models, low
resolutions, or short videos, communication overhead may exceed the compute
savings.

## 6. Memory and output

- Multi-device inference may still replicate the text encoder, VAE, or caches
  on every rank. Do not estimate per-device memory by simply dividing
  single-device memory by the device count. For out-of-memory errors, test CPU
  offload, quantization, and smaller inputs incrementally.
- Keep the model and input configuration identical across ranks. Do not assign
  different random seeds or prompts to individual processes.
- Only the main process should write the final result. Assign a distinct
  `OUTPUT_PATH` to each concurrent job to prevent overwriting.

## 7. Troubleshooting

- **Initialization hangs**: Check the process count, visible device count, and
  product of the parallel sizes. Ensure that no stale process is using the
  rendezvous port, and verify oneCCL and driver compatibility.
- **All-to-all or all-reduce hangs**: Reduce the job to two devices first and
  confirm that the oneCCL variables in the example script have not been
  overridden. Then test pure TP and pure SP separately.
- **One rank exits early**: Inspect the logs from every rank. The earliest
  exception is usually the root cause. Set `PYTHONFAULTHANDLER=1` for more
  complete error information.
- **Incorrect or unstable output**: Establish a baseline with BF16 and the
  repository's default operator settings. Then enable FP8, INT8, custom kernels,
  or compilation one at a time.

Reference scripts:

- `dist_infer/run_minimax_h3_t2av_tp.sh`: 4-device TP
- `dist_infer/run_minimax_h3_t2av_sp_tp.sh`: 8-device SP + TP
- `dist_infer/run_wan22_ti2v_t2v_sp.sh`: 2-device SP
