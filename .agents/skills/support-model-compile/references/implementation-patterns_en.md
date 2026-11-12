# LightX2V Compile Implementation Patterns

English | [简体中文](implementation-patterns.md)

Read this reference when implementing compile support or choosing cache keys. Confirm the paths in the current repository with `rg` before adapting the examples.

## Contents

- [Reference entry points](#reference-entry-points)
- [Shared block integration](#shared-block-integration)
- [Multiple block families](#multiple-block-families)
- [Wan logical layer indices](#wan-logical-layer-indices)
- [Qwen-Image staging identity](#qwen-image-staging-identity)
- [LTX2 and LTX2.3 guider branches](#ltx2-and-ltx23-guider-branches)
- [Lingbot-Video normal MoE](#lingbot-video-normal-moe)
- [Offload integration](#offload-integration)
- [Third-party leaf operators](#third-party-leaf-operators)
- [Compilation inside custom ops and dynamic Triton scalars](#compilation-inside-custom-ops-and-dynamic-triton-scalars)
- [Test skeleton](#test-skeleton)

## Reference entry points

| Purpose | Location |
|---|---|
| Shared block compile and cache | `lightx2v/common/transformer_infer/transformer_infer.py` |
| Wan normal and offload | `lightx2v/models/networks/wan/infer/transformer_infer.py`, `lightx2v/models/networks/wan/infer/offload/transformer_infer.py` |
| Qwen-Image normal and offload | `lightx2v/models/networks/qwen_image/infer/transformer_infer.py`, `lightx2v/models/networks/qwen_image/infer/offload/transformer_infer.py` |
| LTX2 normal, offload, and AR | `lightx2v/models/networks/ltx2/infer/transformer_infer.py`, `lightx2v/models/networks/ltx2/infer/offload/transformer_infer.py`, `lightx2v/models/networks/ltx2/infer/ar_transformer_infer.py` |
| Lingbot-Video normal and MoE | `lightx2v/models/networks/lingbot_video/infer/transformer_infer.py` |
| Qwen kernel guard | `lightx2v/models/networks/qwen_image/weights/transformer_weights.py` |
| Custom-op leaf | `lightx2v/common/ops/rope/flashinfer_rope.py` |
| RMSNorm compile dispatch | `lightx2v/common/ops/norm/rms_norm_weight.py` |
| Platform compile backend | `lightx2v_platform/compilation/ascend_npu/mindie.py`. Shared blocks select the backend from `COMPILE_BACKEND_REGISTER`. |
| Test discovery | `rg --files tests test_cases`. The current snapshot has no committed compile-specific Python tests. |

## Shared block integration

```python
class NewTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        ...
        self.init_compile(config)

    def infer_block(self, block, x, inputs):
        ...
        return x

    def infer_blocks(self, blocks, x, inputs):
        for block_idx, block in enumerate(blocks):
            x = self.run_block(block_idx, block, x, inputs)
        return x
```

The shared `run_block()` handles dispatch.

```python
def run_block(self, block_idx, block, *args):
    if self.use_compile:
        return self.get_compiled_block(block_idx, block)(*args)
    return self.infer_block(block, *args)
```

Keep layer indices out of this dispatcher. Use the shared per-key cache, which also checks object identity on a cache hit. A single callable such as the one below can make all blocks, guiders, and mutable state share guards.

```python
self.compiled_block = torch.compile(self.infer_block, dynamic=None)
```

The shared block cache preserves the `compile_backend` selection. Pass explicit backend arguments through any new dispatcher. Wan and Qwen phase dispatchers currently use the default backend. Validate them separately before extending backend support.

## Multiple block families

Group blocks by signature and semantics. For example, double-stream and single-stream blocks use different methods, return values, and offload managers.

- Use a dispatcher and cache for each family, or include the family in the key.
- Normal inference can use `(family, block_idx)`. Offload without layer-index dependencies can use `(family, id(block))`.
- Keep the two families' `block_idx=0` entries distinct.
- Give different signatures separate compute entry points.
- Use a small model-specific dispatcher until multiple models can share the abstraction.

Check layer state, staging identity, operator compatibility, and warmup for each compilation entry point.

## Wan logical layer indices

Wan attention passes `self.block_idx` downstream. Several subclasses use it for KV cache, adapters, and layer-specific policies.

- Set `self.block_idx` in the owning loop before `run_block()`.
- Cache normal and block offload by `block_idx`.
- Cache phases by `(block_idx, phase_idx)` because the phases read the layer index.
- Supply a stable callable for each phase type.
- Keep specialized post-adapter phases in eager until validated.

Retain the layer-index key even with a small number of staging buffers. Use `id(block)` only after removing the layer-index dependency throughout the graph.

## Qwen-Image staging identity

Qwen-Image blocks and their four phase types are independent of the logical layer index.

- Normal inference uses the default `block_idx`.
- Block offload returns `id(block)`. Cache count equals the number of staging blocks.
- Phase offload caches its four entry points by `phase_idx`.
- Lazy block and phase reuse the same cache.
- The dispatcher requires no `self.block_idx` state.

Qwen-Image currently rejects `use_compile=true` with one-pass RMSNorm. For an `sgl-kernel` config, check the actual compile branch as well as the backend name.

## LTX2 and LTX2.3 guider branches

The reference paths cover normal inference, model offload, and block offload. These are the modes to carry into compile.

- Normal and model offload use the default `block_idx`.
- Block offload uses `id(block)` because its graph is independent of the layer index.
- Convert guider state into a finite set of boolean arguments in the loop.

```python
self.run_block(
    block_idx,
    block,
    vx,
    ax,
    inputs,
    block_idx in self._mm_skip_video_self_blocks,
    block_idx in self._mm_skip_audio_self_blocks,
    self._mm_skip_a2v,
    self._mm_skip_v2a,
)
```

Give `infer_block()` explicit `skip_*` arguments so it can consume booleans without reading changing Python sets.

LTX2 self-attention `cu_seqlens` depends on the request shape. In compile mode, create it before entering the block loop and refresh it from each request's video and audio lengths. This keeps first-block mutation from changing later guards.

LTX2 AR indexes its KV cache by layer. Set `_ar_block_idx` in the AR subclass's `run_block()` before calling `super()`. Keep this field scoped to the AR subclass.

## Lingbot-Video normal MoE

The reference path supports normal inference and reuses the shared `block_idx` cache. Resolve Python configuration such as compute dtype during initialization so the graph can use stable values. MoE's `torch._grouped_mm` remains an external kernel. Validate its workspace and routing workload with representative tokens and steps. Keep compile integration within the existing inference modes.

## Offload integration

For block offload, replace the compute call within the existing stream.

```python
with torch_device_module.stream(self.offload_manager.compute_stream):
    current_block = self.offload_manager.cuda_buffers[0]
    x = self.run_block(block_idx, current_block, x, inputs)
```

Preserve the order `init_first_buffer → prefetch → compute → swap`. Lazy load reuses the staging objects. If a request rebuilds them, the shared cache's identity check refreshes the callable.

Phase offload keeps three responsibilities.

```text
infer_phase         Select and execute the eager phase
get_compiled_phase  Create or reuse the callable by key
run_phase           Dispatch between eager and compile
```

Reuse the existing prefetch and swap loop in both modes.

## Third-party leaf operators

To keep third-party Python outside Dynamo, register the op at module scope.

```python
@torch.library.custom_op(
    "lightx2v::new_op",
    mutates_args=(),
    device_types="cuda",
)
def new_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return external_kernel(x, weight)


@new_op.register_fake
def new_op_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

For in-place mutation, declare the schema and `mutates_args` accurately. Dispatch in `apply()`.

```python
if torch.compiler.is_compiling():
    return new_op(x, weight)
return external_kernel(x, weight)
```

Use minimal inputs to validate numbers, dtype, shape, and the actual kernel. A custom op retains its kernel calls. Use a larger fused op when launch count needs to fall.

Choose the smallest complete semantic unit containing the data-dependent intermediates. If the block map and incremental LUT would make Dynamo trace data dependencies, include them and the attention kernel that consumes the LUT in the same boundary. The fake implementation describes the output metadata visible at the boundary.

## Compilation inside custom ops and dynamic Triton scalars

Dynamo sees one custom op, and its function body still runs. Triton, CUDA extensions, and third-party wrappers inside it have their own JIT and cache. Inspect inner compilation logs or cache artifacts before and after service readiness, then rerun the full lifecycle from a cold cache. Use this evidence alongside `TORCH_LOGS` to determine whether inner compilation occurred during a formal request.

Compare signatures in the order used by the multimodal path.

```text
Original modality tokens
→ packing / padding / alignment
→ TP / SP partition
→ actual q/k lengths on each rank
→ sparse block count and leaf-kernel scalar arguments
```

For variable prompt lengths, first determine which Triton scalars must be static.

- Scalars used only in pointer offsets, masks, or runtime arithmetic can be runtime arguments when the launch grid is computed outside the JIT. Keep static metadata such as block size as `tl.constexpr`.
- Scalars that determine `tl.arange` bounds, static tensor shapes, constexpr branches, unrolling, layout, or `num_warps` need a different parameter split or explicit, finite bucketing and padding.
- For a verified runtime length, use `do_not_specialize` to avoid variants based on runtime values and alignment.

```python
@triton.jit(do_not_specialize=("seq_len",))
def kernel(x, seq_len, block: tl.constexpr):
    ...
```

Apply the decorator only to verified scalars. Validate numerics at two or more distinct lengths, including a nondivisible tail. Check that no new inner compilation artifacts appear after readiness and that steady-state performance holds.

## Test skeleton

Mock `torch.compile` to return the original function when testing Python cache semantics.

```python
with patch.object(common_module.torch, "compile", side_effect=lambda fn, dynamic=None: fn) as compile_mock:
    infer.run_block(0, block, *args)
    infer.run_block(0, block, *args)

assert compile_mock.call_count == 1
```

Cover the default logical-layer key, staging identity, a changed object at the same key, eager bypass, the phase cache, guider branches, and lazy prefetch/swap order. Unit tests verify dispatch. Real-model tests verify Dynamo behavior, kernels, numerical agreement, and performance.
