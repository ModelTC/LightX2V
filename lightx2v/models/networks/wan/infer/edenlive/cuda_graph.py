"""One CUDA graph for the resident EdenLive transformer block stack.

The caller prepares its existing history storage before ``run`` and commits the
returned raw K/V afterwards. Only the current activations and static text K/V
are duplicated; the much larger history storage remains owned by the caller.
"""

import math
from collections.abc import Mapping
from dataclasses import replace

import torch


def _map_tensors(value, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if isinstance(value, Mapping):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported CUDA graph input type: {type(value).__name__}")


def _tensor_metadata(value):
    return str(value.device), value.dtype, tuple(value.shape), tuple(value.stride())


def _tree_signature(value, *, pointers):
    if torch.is_tensor(value):
        metadata = _tensor_metadata(value)
        return ("tensor", metadata, value.data_ptr()) if pointers else ("tensor", metadata)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_tree_signature(item, pointers=pointers) for item in value))
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _tree_signature(item, pointers=pointers)) for key, item in value.items()))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", type(value), value)
    raise TypeError(f"Unsupported CUDA graph input type: {type(value).__name__}")


def _copy_tree(destination, source):
    if torch.is_tensor(destination):
        destination.copy_(source)
    elif isinstance(destination, (tuple, list)):
        for target, value in zip(destination, source):
            _copy_tree(target, value)
    elif isinstance(destination, Mapping):
        for key in destination:
            _copy_tree(destination[key], source[key])


class EdenLiveCUDAGraphStack:
    """Capture and replay one shape of an already compiled transformer stack.

    ``runner(x, embed0, cross_kv)`` must return every tensor needed after replay,
    including raw per-layer K/V used by a graph-external cache commit. Returned
    tensors belong to the graph and are overwritten by its next replay.

    ``bindings`` lists external tensors read or written by the runner (history
    and RoPE). Their storage is retained, never copied. A binding pointer or
    metadata change, a new ``key``, or changed input shapes invalidates the graph.
    The caller's key must also describe non-tensor configuration and weights.

    ``mutated_buffers`` are views of external history regions that the stack
    writes. They are saved during capture and restored between warmups and before
    the first real replay, so capture does not advance or contaminate history.
    Pass only the changed tail, rather than entire history allocations.

    Text K/V is copied into small static allocations. Change ``cross_version``
    whenever its contents change, even when the source pointer is reused. With
    ``cross_version=None``, source pointer changes trigger the refresh instead.
    """

    def __init__(self, warmup_iterations=2):
        if warmup_iterations < 1:
            raise ValueError("CUDA graph capture requires at least one warmup")
        self.warmup_iterations = warmup_iterations
        self.capture_count = 0
        self.replay_count = 0
        self.cross_refresh_count = 0
        self.clear()

    def clear(self):
        """Release graph allocations and external storage references."""
        self.graph = None
        self.outputs = None
        self.static_x = None
        self.static_embed0 = None
        self.static_cross_kv = None
        self._signature = None
        self._cross_version = None
        self._bindings = None
        self._runner = None

    @staticmethod
    def _validate_cuda_inputs(x, embed0, cross_kv, bindings, mutated_buffers):
        if not x.is_cuda or embed0.device != x.device:
            raise ValueError("CUDA graph inputs must be on the same CUDA device")

        def validate(tensor):
            if tensor.device != x.device:
                raise ValueError("CUDA graph tensors must be on the same CUDA device")
            return tensor

        _map_tensors((cross_kv, bindings, mutated_buffers), validate)

    @torch.no_grad()
    def run(self, key, runner, x, embed0, *, cross_kv=(), cross_version=None, bindings=(), mutated_buffers=()):
        """Run one stack, capturing on the first compatible call."""
        self._validate_cuda_inputs(x, embed0, cross_kv, bindings, mutated_buffers)
        signature = (
            key,
            _tensor_metadata(x),
            _tensor_metadata(embed0),
            _tree_signature(cross_kv, pointers=False),
            _tree_signature(bindings, pointers=True),
            _tree_signature(mutated_buffers, pointers=True),
        )
        version = cross_version if cross_version is not None else _tree_signature(cross_kv, pointers=True)
        with torch.cuda.device(x.device):
            if self.graph is None or signature != self._signature:
                self.clear()
                self._capture(signature, runner, x, embed0, cross_kv, version, bindings, mutated_buffers)
            elif version != self._cross_version:
                _copy_tree(self.static_cross_kv, cross_kv)
                self._cross_version = version
                self.cross_refresh_count += 1
            self.static_x.copy_(x)
            self.static_embed0.copy_(embed0)
            self.graph.replay()
            self.replay_count += 1
            return self.outputs

    def _capture(self, signature, runner, x, embed0, cross_kv, cross_version, bindings, mutated_buffers):
        self.static_x = x.clone(memory_format=torch.preserve_format)
        self.static_embed0 = embed0.clone(memory_format=torch.preserve_format)
        self.static_cross_kv = _map_tensors(cross_kv, lambda tensor: tensor.clone(memory_format=torch.preserve_format))
        snapshots = _map_tensors(mutated_buffers, lambda tensor: tensor.clone(memory_format=torch.preserve_format))
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)
        captured = None
        try:
            with torch.cuda.stream(stream):
                for _ in range(self.warmup_iterations):
                    self.static_x.copy_(x)
                    self.static_embed0.copy_(embed0)
                    _copy_tree(mutated_buffers, snapshots)
                    # Drop warmup outputs before capture so their allocations can
                    # be reclaimed instead of pinning one extra full raw-K/V set.
                    outputs = runner(self.static_x, self.static_embed0, self.static_cross_kv)
                    del outputs
                self.static_x.copy_(x)
                self.static_embed0.copy_(embed0)
                _copy_tree(mutated_buffers, snapshots)
            stream.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                outputs = runner(self.static_x, self.static_embed0, self.static_cross_kv)
            current.wait_stream(stream)
        finally:
            # Also restore history after an unsuccessful warmup/capture. The
            # exception propagates so the caller can choose an eager fallback.
            current.wait_stream(stream)
            _copy_tree(mutated_buffers, snapshots)
        self.graph = captured
        self.outputs = outputs
        self._signature = signature
        self._cross_version = cross_version
        self._bindings = bindings
        self._runner = runner
        self.capture_count += 1
        self.cross_refresh_count += 1


class EdenLiveARGraphMixin:
    """Graph eligibility and execution for the fixed-window AR block stack."""

    def _can_graph_stack(self, pre):
        ar = self.config["ar_config"]
        frames, height, width = pre.grid_sizes.tuple
        return (
            self.cuda_graph_stack is not None
            and self.use_compile
            and self.compile_backend == "default"
            and self.config.get("dit_quant_scheme") in ("fp8-sgl", "int8-convrot")
            and self.reuse_history
            and self.reuse_quantized_cross_kv
            and not self.seq_parallel
            and not self.config.get("tensor_parallel", False)
            and not self.config.get("lazy_load", False)
            and not self.has_image_context
            and not self.config.get("lora_configs")
            and not self.config.get("lora_dynamic_apply", False)
            and pre.adapter_args.get("text_segments") is None
            and pre.x.is_cuda
            and frames == ar["num_frame_per_chunk"]
            and self._visible_frames(frames) == ar["local_attn_size"]
            and (height, width) == tuple(size // stride // patch for size, stride, patch in zip(self.config["size"], self.config["vae_stride"][1:], self.config["patch_size"][1:]))
        )

    def infer_without_offload(self, blocks, x, pre_infer_out):
        self._commit_kv_only_active = self._can_commit_kv_only(blocks, pre_infer_out)
        graph_blocks = blocks[:-1] if self._commit_kv_only_active else blocks
        if not graph_blocks or not self._can_graph_stack(pre_infer_out):
            return super().infer_without_offload(blocks, x, pre_infer_out)
        block_key = tuple(id(block) for block in blocks)
        if self._graph_weight_key != block_key:
            # Dynamic adapters must use the ordinary compiled path. Weight
            # reloads explicitly invalidate the graph in EdenLiveModel.
            leaves = [leaf for block in blocks for _, leaf in block.named_weight_leaves()]
            if any(getattr(leaf, "has_lora_branch", False) or getattr(leaf, "has_diff", False) for leaf in leaves):
                return super().infer_without_offload(blocks, x, pre_infer_out)
            self._graph_weight_key = block_key
        grid = pre_infer_out.grid_sizes.tuple
        caches = tuple(self.prepare_self_attn_cache(block.compute_phases[0], x, grid, index) for index, block in enumerate(graph_blocks))
        branch = self.scheduler.infer_condition
        cross_version, cross_kv = self.cross_kv_cache[branch]
        cross_kv = cross_kv[: len(graph_blocks)]
        functions = tuple(self.get_compiled_block(index, block) for index, block in enumerate(graph_blocks))

        def run_stack(static_x, static_embed0, static_cross_kv):
            pre = replace(pre_infer_out, x=static_x, embed0=static_embed0)
            previous = self.current_pre
            self.current_pre = pre
            raw = []
            try:
                for function, cross, cache in zip(functions, static_cross_kv, caches):
                    static_x, k, v = function(static_x, pre, cross, cache)
                    raw.append((k, v))
                return static_x, tuple(raw)
            finally:
                self.current_pre = previous

        tokens = math.prod(grid)
        mutated = tuple((keys[-tokens:], values[..., (keys.shape[0] - tokens) // 16 * 16 :] if self.transposed_v_cache else values[-tokens:]) for keys, values in caches)
        x, raw = self.cuda_graph_stack.run(
            (
                block_key,
                grid,
                branch,
                self.config.get("dit_quant_scheme"),
                self.config["self_attn_1_type"],
                self._commit_kv_only_active,
            ),
            run_stack,
            x,
            pre_infer_out.embed0,
            cross_kv=cross_kv,
            cross_version=(branch, getattr(self, "_graph_cross_generation", 0), cross_version),
            bindings=(caches, self.query_freqs),
            mutated_buffers=mutated,
        )
        if self.scheduler.cache_update:
            pool = self.cache_pools[branch]
            for index, (k, v) in enumerate(raw):
                self.commit_kv(pool, index, k, v, grid)
        if self._commit_kv_only_active:
            last = len(blocks) - 1
            self.block_idx = last
            x = self.run_block(last, blocks[last], x, pre_infer_out)
        return x
