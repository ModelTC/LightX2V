"""One CUDA graph for the resident EdenLive transformer block stack.

The caller prepares its existing history storage before ``run`` and commits the
returned raw K/V afterwards. Only the current activations and static text K/V
are duplicated; the much larger history storage remains owned by the caller.
"""

from collections.abc import Mapping

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
