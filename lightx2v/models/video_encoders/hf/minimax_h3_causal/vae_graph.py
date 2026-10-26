import torch


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
        self.entry = None

    def clear(self):
        if self.entry is not None:
            torch.cuda.synchronize(self.device)
            self.entry = None

    @torch.no_grad()
    def __call__(self, value):
        key = (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
        if self.entry is None or self.entry[0] != key:
            # Keep only the current tile shape, including its graph memory pool.
            self.clear()
            stable = value.clone()
            capture_stream = torch.cuda.Stream(device=value.device)
            capture_stream.wait_stream(torch.cuda.current_stream(value.device))
            with torch.cuda.stream(capture_stream):
                self.decoder(stable)
                self.decoder(stable)
            torch.cuda.synchronize(value.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                output = self.decoder(stable)
            self.entry = key, graph, stable, output, capture_stream
        _key, graph, stable, output, _capture_stream = self.entry
        stable.copy_(value)
        graph.replay()
        # Several local tiles coexist until gather; replay overwrites output.
        return output.clone()
