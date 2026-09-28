# Validation and Reporting Results

English | [简体中文](validation.md)

Select checks according to the change. Documentation or simple script edits do not automatically require full model execution. When storage, copying, or scheduling changes, validate data and lifetimes before generation comparisons. Run the commands below from the repository root in a Python environment matching the target device.

## Entry points and effective configuration

Inspect the model, configuration, task, paths, and environment variables actually passed by each launcher. `scripts/base/base.sh` can override environment settings supplied before it is sourced, especially debugging/timing settings. Read its current contents and explicitly set values after sourcing when needed. Current Qwen layout scripts set BF16, sensitive-layer precision following the main dtype, and `PROFILING_DEBUG_LEVEL=0` after sourcing.

Shell syntax checks, JSON parsing, and replacing the Python inference process with an argument recorder can validate command construction, paths with spaces, working-directory independence, and environment overrides. They do not establish successful model execution. Do not create a permanent testing framework for these low-impact checks.

Within a platform, the two configurations should differ only where needed to select the layout. If weights, precision, inputs, or operators differ, restore comparable conditions or explicitly state that the experiment cannot attribute results to contiguous layout.

## Storage and transfers

Reuse existing tests and add coverage for behavior affected by the change:

| File | Existing coverage |
|---|---|
| [test_block_buffer.py](../../../../test_cases/test_block_buffer.py) | External storage boundaries, address alignment, transposed views, pinning, owner lifetimes, asynchronous double buffering, and auxiliary state |
| [test_block_offload_groups.py](../../../../test_cases/test_block_offload_groups.py) | Wan/Qwen loading versus baseline, ordinary scheduling, multiple groups, original checkpoint dtype, undescribed state, cleanup, and platform import isolation |
| [test_platform_norm_storage.py](../../../../test_cases/test_platform_norm_storage.py) | Floating Norm template inheritance, parameter-free state, precision, and slot reuse |
| [test_platform_quant_storage.py](../../../../test_cases/test_platform_quant_storage.py) | Per-channel quantization templates, scale/bias precision, transpose orientation, checkpoint consumption, double buffering, and native-platform MM comparisons |
| [test_platform_weight_copy.py](../../../../test_cases/test_platform_weight_copy.py) | Platform copy hooks, transposed host views, repeated round trips, and surrounding-memory guards |

Existing CUDA test command:

```bash
PLATFORM=cuda CUDA_VISIBLE_DEVICES=0 \
DTYPE=BF16 SENSITIVE_LAYER_DTYPE=None PROFILING_DEBUG_LEVEL=0 \
python -m pytest -q \
  test_cases/test_block_buffer.py test_cases/test_block_offload_groups.py \
  test_cases/test_platform_norm_storage.py test_cases/test_platform_quant_storage.py \
  test_cases/test_platform_weight_copy.py
```

For NPU, first inspect platform operators in the fixtures and test skip conditions, then run applicable cases with `PLATFORM=ascend_npu` and `ASCEND_RT_VISIBLE_DEVICES=0`. Label NPU operator storage-contract checks performed on CUDA as substitute-device tests. Tests limited to projection or copying do not establish execution of a complete NPU block.

For MLU, replace the first command line with `PLATFORM=cambricon_mlu MLU_VISIBLE_DEVICES=0 \`; for MUSA, use `PLATFORM=musa MUSA_VISIBLE_DEVICES=0 \`. Run in environments with the corresponding vendor runtime installed. Tests select devices through the platform registry; MUSA retains the existing torchada compatibility path. Quantization tests execute a vendor's native kernel only on its own platform. CUDA can substitute for storage and copy checks; other platforms skip foreign-vendor device cases to avoid requiring unrelated quantization dtypes. Passing substitute-device tests does not validate native MLU/MUSA computation.

Key invariants:

- Every planned CPU tensor belongs to the target block's storage, has the planned address/offset, dtype, shape, and stride, and is pinned. Reject out-of-bounds access and unconsumed entries explicitly.
- Baseline tensors can happen to be adjacent because of allocator behavior. Establish contiguous layout from shared storage and the complete layout, not just neighboring pointers. A single tensor's `is_contiguous()` does not establish whole-block contiguity.
- Both device slots retain correct values and stable addresses after repeated H2D transfers. Auxiliary state follows its block, and the main CPU sources remain unchanged during inference.
- Cover the first and last blocks, wraparound, the next step, CFG branches, and, where applicable, successive requests and switching between heterogeneous groups.
- Reject incorrect checkpoint precision/shapes, missing or out-of-scope prefixes, duplicate ownership, and unsupported operators before consuming weights. Binding must not escape the planned storage.
- Close transfers for all groups, not just the currently selected group.

For copies to CPU on NPU, separately check contiguous and transposed/noncontiguous host destinations against original weights after multiple round trips. Data movement is expected to preserve values bit for bit; do not hide copy corruption by relaxing numerical tolerances.

## Real inference comparisons

On the same device and environment, fix checkpoint, dtype, seed, input, resolution/frame count, sampling steps, CFG, operators, and component offload. Use the production baseline and contiguous paths. Substitute kernels or custom inference loops are insufficient for final acceptance.

1. Establish that baseline generation works. If both variants fail, investigate their common loading, computation, and VAE paths.
2. Start with an appropriate number of real blocks/steps for a smoke test. Resolve data and scheduling issues before testing the target scale.
3. Compare relevant intermediate/final latents and decoded outputs. Prefer exact comparisons for deterministic paths. Where nondeterminism exists, repeat the same baseline to establish its variability before explaining tolerances and quality judgments.
4. After changing common code or platform dispatch, retain regression evidence for CUDA's default behavior. Instantiating a new NPU configuration is insufficient.

Report each platform separately, with a concrete scope and statuses such as unverified, storage tests passed, partial real inference passed, or complete generation passed. Records should include the code version, a diff or checksum information for uncommitted changes, device, software versions, configuration, checkpoint, inputs, execution scope, and result location. Do not hardcode expected passing test counts; an entirely skipped suite is not a successful validation.

Historical results apply only to the version tested. If no NPU is available, state that there are no real 910B results for this change. Provide commands for the target machine and the outputs to check. Do not invent successful validation or turn current restrictions into permanent unsupported status.

## Performance and memory, only when requested

Analyze benefits after establishing correctness. Use matching conditions, warmup, and repeated measurements. Report end-to-end inference, denoising, H2D submission time, and device transfer intervals separately. State whether profiling is enabled and account for its overhead.

- Contiguous offload has no inference-time packing. Do not add staging/packing back merely to simplify a comparison.
- Device event intervals can include host submission gaps or waits; they are not necessarily pure DMA time. Nested or overlapping computation, transfer, and host-wait measurements cannot simply be added.
- Count bytes actually copied, distinguishing block payload, padding, auxiliary state, and multiple CFG traversals. State whether reported bandwidth uses decimal (`GB/s`) or binary units.
- Record peak CPU checkpoint-loading memory, persistent pinned storage, device slots, and other component memory separately. Fewer individual allocations do not imply fewer weight bytes.
- CPU cache state after packing is a hypothesis to test in a specific experiment. Slower H2D alone does not establish stale DRAM, a cache-coherence error, or a fixed hardware bottleneck.

Do not automatically add timing code to production inference or restore benchmark infrastructure the user removed. Preserve evidence requested by the user and distinguish measurements from explanations that remain hypotheses.
