from dataclasses import dataclass

import torch


@dataclass
class GraphEntry:
    key: tuple
    graph: torch.cuda.CUDAGraph
    input: torch.Tensor
    output: torch.Tensor
    capture_stream: torch.cuda.Stream


class MiniMaxH3VideoDecoderGraph:
    """Tile CUDA Graph and stream used by causal vae_decode_overlap requests."""

    def __init__(self, vae):
        if vae.execution_device.type != "cuda" or vae.cpu_offload or not vae.decode_parallel:
            raise ValueError("vae_decode_overlap requires CUDA, vae_cpu_offload=false and parallel VAE decoding")
        if vae.decoder.block_offload_runner is not None:
            raise ValueError("vae_decode_overlap requires resident VAE decoder blocks")
        self.decoder = vae.decoder
        self.device = vae.execution_device
        self.stream = torch.cuda.Stream(device=self.device)
        self.entry: GraphEntry | None = None

    def clear(self):
        if self.entry is not None:
            torch.cuda.synchronize(self.device)
            self.entry = None

    @torch.no_grad()
    def __call__(self, value, *, frame_indices):
        key = (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
        if self.entry is None or self.entry.key != key:
            # Keep only the current tile shape, including its graph memory pool.
            self.clear()
            stable = value.clone()
            capture_stream = torch.cuda.Stream(device=value.device)
            capture_stream.wait_stream(torch.cuda.current_stream(value.device))
            with torch.cuda.stream(capture_stream):
                self.decoder(stable)
                self.decoder(stable)
            graph = torch.cuda.CUDAGraph()
            # The graph context synchronizes warmup work before capture.
            with torch.cuda.graph(graph, stream=capture_stream):
                output = self.decoder(stable)
            self.entry = GraphEntry(key=key, graph=graph, input=stable, output=output, capture_stream=capture_stream)
        entry = self.entry
        entry.input.copy_(value)
        entry.graph.replay()
        # Several local tiles coexist until gather; replay overwrites output.
        return entry.output.index_select(2, frame_indices)
