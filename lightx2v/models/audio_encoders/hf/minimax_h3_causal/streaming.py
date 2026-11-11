"""Overlap-save waveform decode matching the H3 v0.1.0 reference."""

import math

import torch

from lightx2v.models.networks.minimax_h3.packing import unpack_audio_tokens


def decoder_context_latents(audio_vae) -> int:
    """Conservative input receptive-field radius of the BigVGAN decoder."""
    decoder = audio_vae.decoder

    def conv_radius(conv, step: float) -> float:
        kernel = int(conv.kernel_size[0])
        dilation = int(conv.dilation[0])
        return (kernel - 1) * dilation * step / 2

    def activation_radius(activation, step: float) -> float:
        upsample = activation.upsample
        downsample = activation.downsample.lowpass
        ratio = int(upsample.stride)
        up_kernel = int(upsample.filter.shape[-1])
        down_kernel = int(downsample.filter.shape[-1])
        # Treat transposed convolution as a full-resolution convolution.
        # This deliberately overestimates its dependency radius.
        return (up_kernel - 1) * step / 2 + (down_kernel - 1) * step / ratio / 2

    radius = conv_radius(decoder.conv_pre, 1.0)
    step = 1.0
    for stage, wrapped_upsampler in enumerate(decoder.ups):
        upsampler = wrapped_upsampler[0]
        radius += conv_radius(upsampler, step)
        step /= int(upsampler.stride[0])
        stage_radius = 0.0
        offset = stage * decoder.num_kernels
        for block in decoder.resblocks[offset : offset + decoder.num_kernels]:
            block_radius = 0.0
            for conv1, conv2, act1, act2 in zip(
                block.convs1,
                block.convs2,
                block.activations[::2],
                block.activations[1::2],
                strict=True,
            ):
                block_radius += activation_radius(act1, step)
                block_radius += conv_radius(conv1, step)
                block_radius += activation_radius(act2, step)
                block_radius += conv_radius(conv2, step)
            stage_radius = max(stage_radius, block_radius)
        radius += stage_radius
    radius += activation_radius(decoder.activation_post, step)
    radius += conv_radius(decoder.conv_post, step)
    # One additional latent covers asymmetric padding/cropping roundoff.
    return math.ceil(radius) + 1


class MiniMaxH3StreamingAudioDecoder:
    """Exact overlap-save decoding for MiniMax-H3's non-causal audio VAE."""

    def __init__(self, audio_vae, *, num_audio_latents: int, decode_graph=None) -> None:
        self.audio_vae = audio_vae
        self.decode_graph = decode_graph
        self.pending = None
        self.num_audio_latents = num_audio_latents
        self.context_latents = decoder_context_latents(audio_vae)
        self.hop_length = int(audio_vae.hop_length)
        self._latents: torch.Tensor | None = None
        self._buffer_start = 0
        self._received = 0
        self._emitted = 0

    def _decode_until(self, end: int) -> tuple[int, torch.Tensor]:
        audio = self.audio_vae.decode_raw(self._latents, stereo_batch=True, return_cpu=False) if self.decode_graph is None else self.decode_graph(self._latents)
        expected_samples = self._latents.shape[-1] * self.hop_length
        if audio.shape[-1] != expected_samples:
            raise RuntimeError(f"audio VAE decode length mismatch: {audio.shape[-1]} != {expected_samples}")
        local_start = (self._emitted - self._buffer_start) * self.hop_length
        local_end = (end - self._buffer_start) * self.hop_length
        start_sample = self._emitted * self.hop_length
        output = audio[..., local_start:local_end]
        self._emitted = end

        keep_start = max(0, self._emitted - self.context_latents)
        drop = keep_start - self._buffer_start
        if drop:
            self._latents = self._latents[..., drop:]
            self._buffer_start = keep_start
        return start_sample, output

    def submit(self, audio_rows):
        """Decode already generated latents while DiT commits clean history."""
        stream = self.decode_graph.stream
        stream.wait_stream(torch.cuda.current_stream(audio_rows.device))
        audio_rows.record_stream(stream)
        with torch.cuda.stream(stream):
            outputs = self.push(audio_rows)
            ready = torch.cuda.Event()
            ready.record()
        self.pending = outputs, ready

    def flush(self):
        """Make the submitted chunk's audio available on the consumer's CUDA stream."""
        outputs, ready = self.pending
        current = torch.cuda.current_stream(self.decode_graph.device)
        current.wait_event(ready)
        # finish() decodes the retained tail on this consumer stream.
        self._latents.record_stream(current)
        for _, audio in outputs:
            audio.record_stream(current)
        self.pending = None
        return outputs

    def close(self):
        """Release request tensors after the runner has synchronized the device."""
        self.pending = None
        self._latents = None

    @torch.no_grad()
    def push(self, audio_rows: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        length = audio_rows.shape[0] // 2
        if length < 1 or self._received + length > self.num_audio_latents:
            raise ValueError("invalid streaming audio chunk length")
        latents = self.audio_vae.denormalize_latents(unpack_audio_tokens(audio_rows, length))
        self._latents = latents if self._latents is None else torch.cat((self._latents, latents), dim=-1)
        self._received += length
        stable_end = max(0, self._received - self.context_latents)
        if stable_end <= self._emitted:
            return []
        return [self._decode_until(stable_end)]

    @torch.no_grad()
    def finish(self) -> tuple[int, torch.Tensor]:
        if self._received != self.num_audio_latents:
            raise RuntimeError(f"received {self._received} audio latents, expected {self.num_audio_latents}")
        output = self._decode_until(self.num_audio_latents)
        self._latents = None
        return output
