---
name: support-static-block-residency
description: Integrate, review, and debug static block residency in LightX2V. Use for standalone compute/H2D measurement, direct resident-count estimation, config generation, uniform placement, event-driven double buffering, cleanup, and baseline comparisons. Deliver CUDA/NPU launchers and validation within the requested scope without automatically adding shared CPU weights or runtime tuning.
---

# LightX2V Static Block Residency Integration

English | [简体中文](SKILL.md)

## Purpose and scope

Reuse the model's production block computation. In a standalone tool, measure representative block compute time `c` and CPU-to-device transfer time `d`, directly estimate resident count `r`, and write it into an inference config. Inference distributes resident blocks uniformly and streams the remaining blocks through two device slots, aiming to reduce transfer waits with as little additional device memory as possible.

The current implementation targets DiT blocks. CPU master weights remain available for every block; selected blocks hold additional device weights for one denoising phase and release them afterward. Text encoders, VAEs, non-block weights, and activations do not automatically adopt this strategy. `r=0` means no additional resident blocks; device buffers are still needed for computation.

Respect the requested scope: analysis requests receive a design; local fixes, explanations, and launcher changes address their respective parts. Complete integration covers estimation, configs, the inference call chain, launchers, and validation. Discover and reuse existing implementations without automatically adding shared weights, parallelism, compile support, model downloads, commits, or pushes.

## Read according to the task

- Changing estimation, config output, residency management, or scheduling: read [implementation patterns](references/implementation-patterns_en.md).
- Reusing Wan or Qwen Image blocks, inputs, or launchers: read the relevant part of [model adapters](references/model-adapters_en.md).
- Selecting tests, measuring performance, or diagnosing problems: read [validation and debugging](references/validation-and-debugging_en.md).

If the model lacks reliable CPU block offload, consult the private-block, weight-binding, and lifetime sections of [support-cpu-block-offload](../support-cpu-block-offload/SKILL_EN.md). Shared CPU weights and host/NUMA handling are separate tasks; do not inherit its complete shared-offload deliverables. Read [support-contiguous-block-offload](../support-contiguous-block-offload/SKILL_EN.md) when the contiguous layout itself needs changes.

Source links are relative to each Markdown file. Check current functions rather than relying on historical line numbers or old experiment conclusions.

## Integration workflow

1. Trace the user's launcher through effective config, runner, model, weights, infer, and manager. Record device, task, checkpoint, precision, operators, input size, CPU weight sources, and existing slots.
2. Establish a working block-offload baseline and determine whether one representative layer reasonably estimates block size and computation. Analyze applicability first for heterogeneous layers, MoE, or substantially different execution paths; do not silently assume homogeneity.
3. Extend the existing [estimator](../../../tools/estimate_offload_residency.py) with a model case: load one real block and construct synthetic inputs matching production dimensions, dtype, positional encoding, and state structure. Reuse common sampling and count calculation rather than adding a model-specific searcher.
4. Generate a config containing `offload_resident_blocks` and a separate measurement report. Inference reads the config and reuses [ResidentBlocks](../../../lightx2v/common/offload/resident.py) and the [event-slot manager](../../../lightx2v/common/offload/event_manager.py), calling production `run_block()` through a model callback.
5. Connect cleanup after denoising, warmup, and exceptional exits. Retain CPU sources so the next request can rebuild resident weights. Configurations that do not select the new path retain existing behavior.
6. Deliver simple launchers, a zero-residency baseline, and bilingual instructions for the requested platforms. Run checks appropriate to the change and distinguish implementation status from measured coverage.

## Essential constraints

- Calculate `r` directly in the standalone tool. Inference does not enumerate candidates, calibrate overlap, search online, or adjust the count per step. Implementation validation does not become a search stage in the estimator.
- Current sampling measures compute and transfer separately, using 3 warmups and 10 timed samples for each and taking the median. Use device events and exclude input resets from the compute interval. Loading and measurement time are distinct from denoising time.
- The aggregate budget `(N-r)*d <= N*c` estimates a count; it does not guarantee that every transfer is hidden with finite double buffering. Zero residency is a valid result when `d <= c`.
- Reuse `uniform_resident_indices()` and retain execution order. Rotate slots according to the actual streamed-block order, rather than substituting the logical layer index modulo the slot count.
- Select the layout independently with `cpu_offload_block_contiguous`: `false` uses per-tensor storage and `true` uses contiguous block storage. Only booleans are accepted; the default is `false`. Match layout and precision between estimation and inference. Remeasure after changes to device, weights, input size, operators, or other measurement conditions.
- Current `ResidentBlocks` relies on `WeightModule` weight leaves and device-transfer interfaces. Check these contracts before integration; do not assume it directly manages arbitrary native `nn.Module` objects.
- Reuse resident weights across denoising steps and release them before VAE decoding instead of rebuilding them each step. See implementation patterns for synchronization, tensor lifetime, slot reuse, and exception recovery.
- Current validation requires single-device eager execution, fixed weights and resolution, and `NoCaching`. Check source validation for shared-weight, parallel, dynamic-adaptation, and other compatibility boundaries. Preserve the user's task and precision; identify required extensions instead of silently removing settings to make a run succeed.
- The tool does not clip `r` to available device memory. Two slots, non-block weights, activations, and other components also consume memory; the count estimate does not guarantee that inference fits.

## Launchers and configs

Use the following structure for complete dual-platform integration. Preserve existing model directories and names; honor requests limited to one platform or a local change.

```text
scripts/<model>/inference/offload_residency/
  estimate.sh
  estimate_npu.sh
  run.sh
  run_npu.sh
  run_baseline.sh
  run_baseline_npu.sh
  README.md
  README_CN.md
configs/<model>/residency/
  baseline.json
  baseline_npu.json
```

Version the base configs. The baseline uses `cpu_offload=true`, `offload_granularity="block"`, and `offload_resident_blocks=0`. The presence of `offload_resident_blocks` selects event-based double buffering, with no extra resident blocks when its value is 0. Omitting the field preserves the legacy path; no separate event switch is needed. Residency launchers consume configs generated under `save_results/`. Do not turn generated configs or reports into fixed hardware answers in repository examples.

Keep project path, model path, and device selection explicit at the top of each launcher. Use `scripts/base/base.sh` and retain full Python commands and output paths. Development-machine absolute paths are editable examples, not general requirements. Do not add layers of environment-variable wrappers, automatic device-count inference, or duplicate task/layout launchers just to standardize a template.

Omit `set -e`, `export PROFILING_DEBUG_LEVEL=0`, and `cd "${lightx2v_path}"` from launchers. Use `"${lightx2v_path}/..."` directly for config, input, output, and `mkdir` paths so launchers preserve the caller's working directory. Inherit profiling from `base.sh`; adjust the shared setting when needed and use the same level for baseline and resident runs.

On each platform, match checkpoint, task, inputs, seed, steps, dtype, operators, layout, and other component offload settings between baseline and residency. The baseline runs independently; residency inference follows estimation for that platform. Users may manually lower `r` in the generated config; rerunning estimation overwrites that change.

Use English `README.md` and Chinese `README_CN.md` with reciprocal language links. Explain assets, editable settings, execution order, output locations, and timing scope. Keep implementation details and historical experiment reports out of the usage guide.

## Validation and delivery

Select checks that cover the change using [validation and debugging](references/validation-and-debugging_en.md). Scheduling or lifetime changes require transfer-count, numerical, repeated-execution, rebuild, and exception checks. Documentation changes need content and link checks.

A complete integration report includes:

1. Model, component, platform, and task scope, with key estimation and inference entry points.
2. Measurement conditions, `c`, `d`, `N`, `r`, actual resident indices, and streamed-block count.
3. Locations of baseline/residency launchers, base/generated configs, and bilingual guides.
4. Correctness and lifetime evidence; first-step, steady denoising, and memory comparisons when performance is requested.
5. Unverified platforms, input sizes, or combinations, with reasons.

When hardware is unavailable, complete feasible implementation and static checks and identify missing device validation. Config parsing, a successful `--help`, or CUDA results do not replace NPU measurements. Small-input checks do not establish that the default target size works.
