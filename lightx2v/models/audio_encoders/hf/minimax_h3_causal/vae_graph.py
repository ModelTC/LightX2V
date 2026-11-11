import math

import torch
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from lightx2v.models.audio_encoders.hf.minimax_h3_causal.streaming import decoder_context_latents
from lightx2v.models.networks.minimax_h3.packing import FPS, FRAMES_PER_CHUNK


class MiniMaxH3AudioDecoderGraph:
    """FP32 audio decode graphs and stream for RefA2AV's generated waveform."""

    @torch.no_grad()
    def __init__(self, vae):
        if vae.execution_device.type != "cuda" or vae.cpu_offload:
            raise ValueError("audio_vae_decode_overlap requires a resident CUDA audio VAE")
        self.vae = vae
        self.device = vae.execution_device
        self.stream = torch.cuda.Stream(device=self.device)
        self.entries = {}
        context = decoder_context_latents(vae)
        chunk = FRAMES_PER_CHUNK * vae.sampling_rate / (FPS * vae.hop_length)
        self.window_lengths = {2 * context + math.floor(chunk), 2 * context + math.ceil(chunk)}
        # Evaluate fixed parametrizations once, on the same device and in FP32.
        # Reference audio encoding retains its original weights and execution.
        for module in vae.decoder.modules():
            if is_parametrized(module, "weight"):
                remove_parametrizations(module, "weight", leave_parametrized=True)

    @torch.no_grad()
    def __call__(self, value):
        # Only the two steady streaming windows are captured. Startup and tail
        # lengths use native decoding, so graph memory does not grow with time.
        if value.shape[-1] not in self.window_lengths:
            return self.vae.decode_raw(value, stereo_batch=True, return_cpu=False)
        key = tuple(value.shape)
        if key not in self.entries:
            stable = value.clone(memory_format=torch.contiguous_format)
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self.vae.decode_raw(stable, stereo_batch=True, return_cpu=False)
            graph = torch.cuda.CUDAGraph()
            # The graph context synchronizes warmup work before capture.
            with torch.cuda.graph(graph, stream=capture_stream):
                output = self.vae.decode_raw(stable, stereo_batch=True, return_cpu=False)
            self.entries[key] = graph, stable, output, capture_stream
        graph, stable, output, _capture_stream = self.entries[key]
        stable.copy_(value)
        graph.replay()
        # The caller may retain a waveform slice across another graph replay.
        return output.clone()

    def clear(self):
        torch.cuda.synchronize(self.device)
        self.entries.clear()
