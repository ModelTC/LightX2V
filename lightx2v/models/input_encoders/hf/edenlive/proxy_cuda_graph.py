"""CUDA graph replay for the stateless, resident ProxyDINO encoder."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class _GraphEntry:
    inputs: torch.Tensor
    stream: torch.cuda.Stream
    warmups: int = 0
    graph: torch.cuda.CUDAGraph | None = None
    output: torch.Tensor | None = None


class ProxyCUDAGraph:
    """Capture an unchanged eager encoder after two calls with the same shape.

    Input storage is staged and outputs are cloned, so a later replay cannot
    overwrite features that a consumer still uses. At most two input shapes
    retain private graph pools; uncommon tail chunks can use the second slot.
    The wrapped forward must be stateless and return one CUDA tensor.
    """

    def __init__(self, forward: Callable[[torch.Tensor], torch.Tensor], *, warmup_iters: int = 2, max_graphs: int = 2):
        if warmup_iters < 1 or max_graphs < 1:
            raise ValueError("Proxy CUDA graph warmup_iters and max_graphs must be positive")
        self.forward = forward
        self.warmup_iters = warmup_iters
        self.max_graphs = max_graphs
        self._entries: OrderedDict[tuple, _GraphEntry] = OrderedDict()

    def clear(self):
        """Drop captures before replacing weights or unloading the encoder."""
        for entry in self._entries.values():
            entry.stream.synchronize()
        self._entries.clear()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if not images.is_cuda or torch.cuda.is_current_stream_capturing():
            return self.forward(images)
        key = (images.device, images.dtype, tuple(images.shape))
        with torch.cuda.device(images.device):
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) == self.max_graphs:
                    _, evicted = self._entries.popitem(last=False)
                    evicted.stream.synchronize()
                entry = _GraphEntry(torch.empty_like(images, memory_format=torch.contiguous_format), torch.cuda.Stream(device=images.device))
                self._entries[key] = entry
            self._entries.move_to_end(key)

            caller_stream = torch.cuda.current_stream(images.device)
            entry.stream.wait_stream(caller_stream)
            images.record_stream(entry.stream)
            with torch.cuda.stream(entry.stream):
                entry.inputs.copy_(images)
                if entry.warmups < self.warmup_iters:
                    output = self.forward(entry.inputs)
                    entry.warmups += 1
                else:
                    if entry.graph is None:
                        # Publish only a successfully captured graph. Lazy
                        # CUDA library initialisation drains during warmup on
                        # this same stream, before capture begins.
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=entry.stream, capture_error_mode="thread_local"):
                            output = self.forward(entry.inputs)
                        entry.output = output
                        entry.graph = graph
                    entry.graph.replay()
                    output = entry.output
                result = output.clone()
            caller_stream.wait_stream(entry.stream)
            result.record_stream(caller_stream)
            return result
