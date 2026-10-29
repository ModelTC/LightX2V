# Estimation, Residency Management, and Scheduling

English | [简体中文](implementation-patterns.md)

## Sources and responsibilities

| Entry point | Responsibility |
|---|---|
| [estimate_offload_residency.py](../../../../tools/estimate_offload_residency.py) | Single-block cases, timing, count calculation, config and report output |
| [resident.py](../../../../lightx2v/common/offload/resident.py) | `uniform_resident_indices()`, common config validation, and `ResidentBlocks` preparation/release |
| [event_manager.py](../../../../lightx2v/common/offload/event_manager.py) | Two fixed slots, ready/free events, and `run_blocks()` in `EventSlotWeightAsyncStreamManager` |
| [weight_module.py](../../../../lightx2v/common/modules/weight_module.py) | Weight-leaf traversal, CPU/device state, and loading interfaces |
| [block_loader.py](../../../../lightx2v/common/offload/block_loader.py) | `OffloadGroup`, contiguous-layout preparation, and config validation |
| [set_config.py](../../../../lightx2v/utils/set_config.py) | `build_startup_config()` merges and normalizes the runtime model config |

## Standalone estimation flow

```text
Base JSON + CLI + effective dtype/operator environment
    → build_startup_config()
    → build_<model>_case()
        → load_block(): one CPU block + two device slots
        → production operators + synthetic inputs with the actual shape
    → measure_block(): measure d and c separately
    → estimate_resident_count(N, c, d)
    → generated config + measurement report
```

`load_block()` selects one block's real weights from `SafetensorsCheckpoint` using the model prefix, applies the production block class, quantization scheme, and runtime dtype conversions, and creates device slots. Contiguous layout uses `prepare_contiguous_group()` and the production transfer path; per-tensor transfer is the default. Random weight matrices, one bare memcpy, or a generic GEMM do not establish equivalent block estimation.

Activations and conditioning inputs are synthetic. Their shapes, dtypes, text lengths, positional encodings, and image/text state must reflect the target config. Estimation loads the required block and model metadata/tokenizer without requiring the entire DiT, text encoder, or VAE to run full inference.

New model cases should reuse the current return contract:

| Field | Use |
|---|---|
| `blocks`, `manager` | Common code transfers CPU sources into a device slot, then synchronizes and cleans up after measurement |
| `compute` | No-argument callback invoking production single-block computation; returns a tensor sequence for finite-value checks |
| `reset` | No-argument callback restoring inputs that may be modified in place; runs before the timing start |
| `block_bytes` | Block weight bytes under the actual loading/layout rules, for reporting |
| `grid`, `pixel_size`, `text_tokens` | Actual measured input size |
| `infer`, `pre` | Model objects and inputs retained by current cases, also used by real-block comparison checks |
| `text_lengths` | Positive/negative prompt lengths for the current Qwen case; record for other models when relevant |

Add the CLI model choice and case dispatch to reuse common measurement. Adjust report branches as needed so a new model does not accidentally inherit Wan frame fields. See [model adapters](model-adapters_en.md) for model-specific construction.

## Measurement scope for c and d

`measure_ms()` uses timing-enabled device Events on the selected stream: prepare inputs → start → action → end → wait for end. Each measurement performs `WARMUP=3` plus `REPEATS=10` iterations, retains only the last 10 times, and takes their median. Input reset precedes the timing start and is excluded from the sample.

`measure_block()` first measures `copy_block_to_buffer()` on the transfer stream to obtain `d`. After synchronization, it measures single-block computation with weights already loaded, on the compute stream, to obtain `c`. It checks computed outputs for finite values. These measurements are independent; they do not measure concurrent contention, run candidate resident sets, or calibrate overlap.

The report retains raw timed samples in `compute_samples_ms` and `transfer_samples_ms`; `compute_ms` and `transfer_ms` are medians. `elapsed_seconds` is the tool's elapsed time up to that stage, including preceding preparation, rather than a substitute for a block measurement or full inference time.

## Count and uniform placement

Let `N` be the total block count and `r` the resident count. Using one block to represent the layers gives compute budget `N*c` and transfer budget `(N-r)*d` for one complete block traversal. Require the latter not to exceed the former:

```text
(N-r)*d <= N*c
r >= N - N*c/d

d <= c: r = 0
d > c : r = N - floor(N*c/d)
```

`estimate_resident_count()` requires a positive integer `N` and finite positive `c` and `d`. Call the existing function, preserving integer rounding and boundary checks. Do not change the current strategy by adding candidate enumeration, empirical safety factors, or automatic memory clipping.

This is the minimum integer under an aggregate-budget approximation, not a sufficient condition for fully hiding transfers in the actual schedule. Finite prefetch windows, startup/drain overhead, layer differences, and concurrent bandwidth contention can leave waits. Inference comparisons establish actual gains; validation does not automatically feed back into another count-selection algorithm.

`uniform_resident_indices(N, r)` uses zero-based indices:

```text
r = 0: ()
r > 0: tuple(floor(k*N/r) for k in range(r))
```

For example, `N=8, r=2` gives `(0, 4)`. There are exactly `r` unique sorted indices, and adjacent circular gaps differ by at most 1. Uniformity describes the selected resident layers; it does not reorder computation or concentrate residency in the first `r` layers.

## Configs and reports

Retain model, operator, and other offload settings from the base config. Preserve `cpu_offload_block_contiguous` during measurement and in the generated config, writing `false` if it was omitted. Derive the report’s `layout` from that boolean. When constructing the runtime config, the tool sets `cpu_offload=true`, `offload_granularity="block"`, and `offload_resident_blocks=0`. After measurement it writes the result into `offload_resident_blocks`. Any obsolete separate event switch in the input is removed from the output config. Wan also saves normalized `num_frames`.

An explicit `--output-config` selects the destination; otherwise the tool writes `<stem>.resident.json` beside the input config. The report uses the output filename with a `.profile.json` suffix. `write_json()` creates a temporary file in the target directory and uses `os.replace()` to replace each JSON file without exposing a partial write.

The report includes `num_blocks` for `N`, `resident_blocks`, `streamed_blocks`, `resident_indices`, input size, device, dtype, quantization, and layout. Inference recomputes indices from the config's count; it does not read the profile to schedule blocks.

For homogeneous blocks, reported `resident_weight_bytes` is approximately `r*block_bytes`, and `execution_buffer_bytes` is `2*block_bytes`. This weight budget excludes activations, non-block weights, operator workspace, other components, and allocator overhead. Even `r=N` does not currently promise to eliminate the two slots automatically. CPU master weights remain available throughout.

## Inference scheduling and lifetime

Wan, Qwen Image, Qwen-Image-2.1, and MiniMax-H3 select the event path when the config contains `offload_resident_blocks`. A value of 0 retains event-based double buffering; a positive value adds that many resident blocks. Omitting the field preserves the legacy path. The `r=0` event baseline and legacy offload are distinct comparisons with different purposes.

`ResidentBlocks.prepare()` idempotently computes indices, saves original weight-leaf state, calls selected blocks' `to_cuda_async()`, and logs resident count, streamed count, and indices. It relies on contracts including `named_weight_leaves()` and `pin_*` CPU masters. Adapt a new model's interfaces when needed instead of claiming integration merely from instantiating the class.

Current `run_blocks()` sequence:

1. Compute and transfer streams wait for the caller stream, covering resident-weight and input preparation.
2. Select nonresident blocks in logical order and prefetch up to two into their respective slots.
3. Traverse all blocks in order: compute resident blocks directly; for other blocks, wait for slot readiness, compute, record free, and queue the next nonresident block into that slot.
4. Slot copies wait for the previous consumer's free event; computation waits for the copy's ready event. Rotate slots only after nonresident blocks.
5. The caller stream waits for computation before receiving output. Exceptions synchronize both streams and reset event bookkeeping.

Model callbacks preserve production semantics, including logical layer indices and state ordering. Qwen carries both text and image state; neither may be dropped. Caller-owned inputs used on another stream need appropriate `record_stream()` calls or equivalent lifetime guarantees. Outputs returned to the caller also need allocator lifetime handling. Stream waits establish execution dependencies, not storage lifetime.

Resident weights are prepared on first use and reused across steps and CFG branches. The runner invokes `release_resident_blocks()` through `finally` after denoising, warmup, or failure, releasing additional device weights before VAE decoding. `ResidentBlocks.release()` waits for device work and restores original attribute references; the model release entry then calls `reset_slots()`. Resetting bookkeeping does not wait for completion and cannot by itself release resources still in use.

The next request prepares from the same CPU masters; verify release → prepare → infer. Do not delete CPU masters to release device weights or rely on incidental GC timing to avoid read/write conflicts.

## Current config boundaries

Common `validate_residency_config()` requires CPU block offload, `AI_DEVICE` of `cuda` or `npu`, and single-device eager execution with fixed weights/resolution and `NoCaching`. It currently rejects combinations including lazy loading, shared CPU weights, parallelism, LoRA/adapters, changing resolution, compile, and CUDA Graphs. Consult the function for exact fields.

Model validators additionally check model identity, task, and conditioning. Preserve existing configuration behavior for new models and apply relevant checks only when the event/residency path is selected. These limits describe current implementation coverage, not fundamental impossibility. Extending compatibility requires implementation and validation; bypassing a guard does not establish support.
