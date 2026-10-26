"""CUDA graph execution for a fixed-shape autoregressive block stack."""

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


def _tensor_layout(tensor):
    return tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())


class CUDAGraphPool:
    """Share temporary allocations and one capture stream between sequential graphs."""

    def __init__(self, device):
        self.handle = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=device)
        self.inputs = {}
        self.input_versions = {}
        self.outputs = {}

    def stage_inputs(self, inputs, input_versions):
        layout = tuple(_tensor_layout(tensor) for tensor in inputs)
        staged = self.inputs.get(layout)
        if staged is None:
            staged = tuple(tensor.clone(memory_format=torch.preserve_format) for tensor in inputs)
            self.inputs[layout] = staged
            self.input_versions[layout] = tuple((version, tensor.data_ptr()) if version is not None else None for tensor, version in zip(inputs, input_versions))
        return staged

    def copy_inputs(self, destination, source, versions):
        # Variants share staged tensors, so compare content tokens pool-wide.
        # Include the source pointer to prevent unrelated callers from using
        # the same token for different buffers with identical layouts.
        layout = tuple(_tensor_layout(tensor) for tensor in source)
        previous = self.input_versions[layout]
        current = tuple((version, tensor.data_ptr()) if version is not None else None for tensor, version in zip(source, versions))
        for target, value, version, old, new in zip(destination, source, versions, previous, current):
            if version is None or new != old:
                target.copy_(value)
        self.input_versions[layout] = current

    def stage_outputs(self, outputs):
        tensors, structure = tree_flatten(outputs)
        layout = structure, tuple(_tensor_layout(tensor) for tensor in tensors)
        staged = self.outputs.get(layout)
        if staged is None:
            staged = tuple(torch.empty_like(tensor, memory_format=torch.preserve_format) for tensor in tensors)
            self.outputs[layout] = staged
        return tree_unflatten(staged, structure)

    @staticmethod
    def copy_outputs(destination, source):
        for target, value in zip(tree_flatten(destination)[0], tree_flatten(source)[0]):
            target.copy_(value)


class AutoregressiveCUDAGraphStack:
    """Keep a block-stack graph per variant while slot and position inputs change.

    ``runner(*inputs)`` receives graph-owned copies of a flat tensor tuple.
    External KV and weight storage stays in ``bindings``; a new pointer or
    layout invalidates capture. ``key`` describes non-tensor execution choices.
    ``variant`` separates denoising and cache-commit captures; counters include both.

    The caller advances CPU cache metadata outside the graph and supplies GPU
    slot indices through a fixed input or binding buffer. ``mutated_buffers``
    contains the physical KV regions written by the first call, restored after
    each warmup and capture. Outputs belong to the graph until its next replay.

    ``input_versions`` optionally supplies one token per input. ``None`` copies
    that input every call; unchanged tokens retain its staged contents. Inputs
    with version tokens must remain read-only in the runner.

    ``memory_pool`` shares scratch storage for sequential block graphs. Their
    outputs must be consumed or staged before another graph uses the same pool.
    Input and output buffers with matching layouts are shared. Inputs with no
    version token are refreshed on every replay; results must be consumed before
    the next graph uses the pool. ``on_capture`` drains asynchronous GPU work
    only when a new graph must be captured.
    """

    def __init__(self, warmup_iterations=2):
        if warmup_iterations < 1:
            raise ValueError("CUDA graph capture needs at least one warmup")
        self.warmup_iterations = warmup_iterations
        self.capture_count = 0
        self.replay_count = 0
        self._variants = {}
        self._clear_capture()

    def clear(self):
        self._clear_capture()
        for stack in self._variants.values():
            stack.clear()
        self._variants.clear()

    def _clear_capture(self):
        self.graph = None
        self.outputs = None
        self.static_inputs = ()
        self._input_versions = ()
        self._signature = None
        self._bindings = ()
        self._runner = None
        self._memory_pool = None

    def _copy_inputs(self, inputs, input_versions):
        if self._memory_pool is not None:
            self._memory_pool.copy_inputs(self.static_inputs, inputs, input_versions)
            return
        for destination, source, version, previous in zip(self.static_inputs, inputs, input_versions, self._input_versions):
            if version is None or version != previous:
                destination.copy_(source)
        self._input_versions = input_versions

    @staticmethod
    def _restore_buffers(buffers, snapshots):
        for destination, source in zip(buffers, snapshots):
            destination.copy_(source)

    @torch.no_grad()
    def run(self, key, runner, inputs, *, bindings=(), mutated_buffers=(), input_versions=None, variant=None, memory_pool=None, on_capture=None):
        """Capture per signature and call ``on_capture`` only before a new capture."""
        if variant is not None:
            stack = self._variants.get(variant)
            if stack is None:
                stack = self._variants[variant] = AutoregressiveCUDAGraphStack(self.warmup_iterations)
            captures, replays = stack.capture_count, stack.replay_count
            try:
                return stack.run(key, runner, inputs, bindings=bindings, mutated_buffers=mutated_buffers, input_versions=input_versions, memory_pool=memory_pool, on_capture=on_capture)
            finally:
                self.capture_count += stack.capture_count - captures
                self.replay_count += stack.replay_count - replays
        # A shared pool tracks versioned contents across variants. Inputs
        # without explicit versions are refreshed on every replay.
        input_versions = (None,) * len(inputs) if input_versions is None else input_versions
        signature = (
            key,
            None if memory_pool is None else memory_pool.handle,
            tuple(_tensor_layout(tensor) for tensor in inputs),
            tuple((_tensor_layout(tensor), tensor.data_ptr()) for tensor in bindings),
        )
        with torch.cuda.device(inputs[0].device):
            if self.graph is None or signature != self._signature:
                if on_capture is not None:
                    on_capture()
                self._clear_capture()
                # A new capture can overwrite a preceding graph's output in the
                # shared pool. Keep its inputs outside that pool during capture.
                if memory_pool is not None:
                    inputs = tuple(tensor.clone(memory_format=torch.preserve_format) for tensor in inputs)
                self._capture(signature, runner, inputs, bindings, mutated_buffers, input_versions, memory_pool)
            self._copy_inputs(inputs, input_versions)
            self.graph.replay()
            self.replay_count += 1
            return self.outputs

    def _capture(self, signature, runner, inputs, bindings, mutated_buffers, input_versions, memory_pool):
        self._memory_pool = memory_pool
        self.static_inputs = tuple(tensor.clone(memory_format=torch.preserve_format) for tensor in inputs) if memory_pool is None else memory_pool.stage_inputs(inputs, input_versions)
        self._input_versions = input_versions
        snapshots = tuple(tensor.clone(memory_format=torch.preserve_format) for tensor in mutated_buffers)
        current_stream = torch.cuda.current_stream(inputs[0].device)
        capture_stream = torch.cuda.Stream(device=inputs[0].device) if memory_pool is None else memory_pool.stream
        capture_stream.wait_stream(current_stream)

        try:
            with torch.cuda.stream(capture_stream):
                for _ in range(self.warmup_iterations):
                    self._copy_inputs(inputs, input_versions)
                    self._restore_buffers(mutated_buffers, snapshots)
                    outputs = runner(*self.static_inputs)
                    if memory_pool is not None:
                        staged_outputs = memory_pool.stage_outputs(outputs)
                    del outputs
                self._copy_inputs(inputs, input_versions)
                self._restore_buffers(mutated_buffers, snapshots)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream, pool=None if memory_pool is None else memory_pool.handle):
                outputs = runner(*self.static_inputs)
                if memory_pool is not None:
                    memory_pool.copy_outputs(staged_outputs, outputs)
                    del outputs
                    outputs = staged_outputs
        finally:
            current_stream.wait_stream(capture_stream)
            self._restore_buffers(mutated_buffers, snapshots)

        self.graph = graph
        self.outputs = outputs
        self._signature = signature
        self._bindings = bindings
        self._runner = runner
        self._memory_pool = memory_pool
        self.capture_count += 1
