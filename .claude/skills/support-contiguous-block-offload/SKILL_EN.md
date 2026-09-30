---
name: support-contiguous-block-offload
description: Integrate, review, and debug cpu_offload_layout=contiguous for LightX2V models. Covers common block declarations, operator storage contracts, NVIDIA CUDA and Ascend NPU isolation, baseline/contiguous entry points, and correctness validation. Intended for persistent contiguous pinned weights, without automatically adding multiprocess weight sharing or runtime packing.
---

# Integrating Contiguous CPU Block Offload

English | [简体中文](SKILL.md)

## Purpose and scope

Reuse the common infrastructure. During loading, write each block's main weights into their final contiguous pinned CPU storage. Create device tensor views with the same layout, transfer the entire block during inference, and preserve the model's existing computation and double-buffer scheduling. Inference does not repeatedly pack weights or allocate staging buffers.

Contiguity refers to virtual addresses within a block. Separate blocks need not be adjacent, and physical pages need not be contiguous. This does not eliminate weight transfers or guarantee faster inference. Auxiliary device state may still require separate copies.

Follow the user's scope. Analysis requests produce proposals only; a local fix does not require repeating a complete model integration. For tasks covering both platforms, check NVIDIA GPU and Ascend NPU entry points and report implementation and validation status separately. Respect tasks limited to one platform. When hardware is unavailable, complete the feasible code and local checks and identify what remains unverified.

This skill focuses on private contiguous weights within each rank. Consult [support-cpu-block-offload](../support-cpu-block-offload/SKILL_EN.md) only when shared weights or host/NUMA replicas are in scope; do not inherit its delivery requirements for a complete shared-weight integration.

## Read according to the task

- Changing block declarations, loading, or scheduling: read [Common integration](references/common-integration_en.md).
- Integrating CUDA or changing code shared by both platforms: read [NVIDIA GPU](references/cuda_en.md).
- Integrating Ascend: read [Ascend NPU](references/ascend-npu_en.md), including CUDA regression requirements when shared code changes.
- Reusing model definitions and configurations: read the relevant sections of [Wan and Qwen examples](references/model-examples_en.md).
- Selecting tests or interpreting results: read [Validation](references/validation_en.md).

Use current source files and function names instead of historical line numbers. Repository links in this skill and its references are relative to the document containing each link.

## Integration workflow

1. Trace the user's launcher through the effective configuration, runner, checkpoint loading, weights, inference, and offload manager. Confirm precision, task, and the scope of components such as the text encoder and VAE.
2. Check whether the model already has native block offload, two device slots, and a stable block definition. Identify CPU blocks, slots, checkpoint prefixes, and the storage contract of each weight operator.
3. Address gaps in the layer that owns them. Models declare blocks; operators describe and bind their state; common code plans layouts and manages lifetimes; platform code handles allocation, copying, and special formats.
4. Validate the ordinary per-tensor block baseline before enabling contiguous storage. If native block offload and operator contracts already exist, prefer adding only block registration. Otherwise, explicitly implement missing contracts; block registration alone does not adapt an arbitrary model.
5. Provide the platform entry points requested by the task and run relevant tests and real inference comparisons. Report changes, validation evidence, restrictions, and any platform validation still outstanding.

## Architecture constraints

- Use [WeightModule.register_offload_group](../../../lightx2v/common/modules/weight_module.py), the common [block_loader.py](../../../lightx2v/common/offload/block_loader.py), and [block_layout.py](../../../lightx2v/common/offload/block_layout.py). Do not copy the former Wan-specific layout loader into a second loading system for another model.
- Each weight container and manager owns one group. CPU blocks and device slots must have compatible layouts. Weight ownership must not overlap, and undescribed weights or stateful operators must not be silently skipped.
- Prefer reusing an operator's `load(BlockLoadContext)`. Add `bind_storage()` only when special binding is needed; avoid wrappers that merely forward calls.
- Operator selection belongs in configuration. Declare device capabilities through the existing platform registration mechanism. Do not add per-chip branches to models or the common loader, or create platform adapter modules specifically for Wan/Qwen.
- Preserve NVIDIA's default operators, precision, and behavior when contiguous layout is disabled. Keep platform-specific handling in `lightx2v_platform` and check CUDA import isolation and numerical regressions.
- Preserve source storage lifetimes, dependencies before overwriting slots, and cleanup of the current transfer. A single successful H2D copy is insufficient validation.

## Delivering both platform paths

For a complete integration on both platforms, use the following directory convention. Preserve user settings in existing files and change only what the task requires:

```text
scripts/<model>/offload_layout/
  run_baseline.sh
  run_contiguous.sh
  run_baseline_npu.sh
  run_contiguous_npu.sh
  README.md
  README_CN.md
configs/<model>/layout/
  baseline.json
  continuous.json
  baseline_npu.json
  continuous_npu.json
```

Within each platform, keep baseline and contiguous variants identical in model, checkpoint, precision, input, seed, sampling, component offload, and compute operators. Prefer adding only `"cpu_offload_layout": "contiguous"`. Scripts use `contiguous`; configuration filenames retain `continuous`.

Keep scripts simple and make model paths, device selection, and output paths explicit. Do not treat development-machine absolute paths as universal requirements. Document actual commands and precision requirements. Handle timing tools, downloads, commits, or pushes only when requested.

Report each platform's entry points, configurations, actual operators, validation scope, and evidence. Distinguish entry points being added, storage/transfer tests passing, complete generation passing, and performance being measured. Test counts or results from older versions do not establish support in the current version.
