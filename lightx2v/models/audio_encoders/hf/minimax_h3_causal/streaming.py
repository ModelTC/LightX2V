"""Overlap-save waveform decode matching the H3 v0.1.0 reference."""

import torch

from lightx2v.models.networks.minimax_h3.packing import unpack_audio_tokens

# decoder_context_latents gives 29 latents on either side plus one of margin.
AUDIO_DECODER_CONTEXT_LATENTS = 30


def decoder_context_latents(audio_vae) -> int:
    """Recalculate the fixed context from the decoder's integer index support.

    Streaming and CUDA Graph setup use the constant above. One output hop
    covers every upsampling phase, independently of the audio content.
    This follows the BigVGAN operator layout; recheck both the traversal and
    the constant when that layout changes.
    """
    decoder = audio_vae.decoder

    def conv_bounds(bounds, conv):
        left, right = bounds
        stride, padding = conv.stride[0], conv.padding[0]
        return left * stride - padding, right * stride - padding + (conv.kernel_size[0] - 1) * conv.dilation[0]

    def transpose_bounds(bounds, kernel, stride, padding=0):
        left, right = bounds
        # Input i writes outputs [i * stride - padding, ... + kernel - 1].
        return -((kernel - 1 - left - padding) // stride), (right + padding) // stride

    def activation_bounds(bounds, activation):
        upsample = activation.upsample
        downsample = activation.downsample.lowpass
        left = bounds[0] * downsample.stride - downsample.pad_left
        right = bounds[1] * downsample.stride - downsample.pad_left + downsample.filter.shape[-1] - 1
        # Undo the upsampler's output crop, transposed convolution and input pad.
        left, right = transpose_bounds((left + upsample.pad_left, right + upsample.pad_left), upsample.filter.shape[-1], upsample.stride)
        return left - upsample.pad, right - upsample.pad

    # The union of one hop's phases (800 for v0.1) can be propagated as one interval.
    bounds = conv_bounds((0, audio_vae.hop_length - 1), decoder.conv_post)
    bounds = activation_bounds(bounds, decoder.activation_post)
    for stage in reversed(range(decoder.num_upsamples)):
        branches = []
        offset = stage * decoder.num_kernels
        for block in decoder.resblocks[offset : offset + decoder.num_kernels]:
            branch = bounds
            layers = zip(block.convs1, block.convs2, block.activations[::2], block.activations[1::2], strict=True)
            for conv1, conv2, act1, act2 in reversed(tuple(layers)):
                residual = activation_bounds(conv_bounds(branch, conv2), act2)
                residual = activation_bounds(conv_bounds(residual, conv1), act1)
                branch = min(branch[0], residual[0]), max(branch[1], residual[1])
            branches.append(branch)
        bounds = min(left for left, _ in branches), max(right for _, right in branches)
        upsampler = decoder.ups[stage][0]
        bounds = transpose_bounds(bounds, upsampler.kernel_size[0], upsampler.stride[0], upsampler.padding[0])
    bounds = conv_bounds(conv_bounds(bounds, decoder.conv_pre), audio_vae.dec_in_proj)
    return max(-bounds[0], bounds[1]) + 1


class MiniMaxH3StreamingAudioDecoder:
    """Overlap-save audio decoding with optional left-context-only live output."""

    def __init__(self, audio_vae, *, num_audio_latents: int | None, decode_graph=None, low_latency: bool = False) -> None:
        self.audio_vae = audio_vae
        self.decode_graph = decode_graph
        self.pending = None
        self.num_audio_latents = num_audio_latents
        self.low_latency = low_latency
        self.context_latents = AUDIO_DECODER_CONTEXT_LATENTS
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
        if length < 1 or (self.num_audio_latents is not None and self._received + length > self.num_audio_latents):
            raise ValueError("invalid streaming audio chunk length")
        latents = self.audio_vae.denormalize_latents(unpack_audio_tokens(audio_rows, length))
        self._latents = latents if self._latents is None else torch.cat((self._latents, latents), dim=-1)
        self._received += length
        if self.low_latency:
            # Live video emits its first two chunks together on the second push.
            if self._received == length:
                return []
            return [self._decode_until(self._received)]
        stable_end = max(0, self._received - self.context_latents)
        if stable_end <= self._emitted:
            return []
        return [self._decode_until(stable_end)]

    @torch.no_grad()
    def finish(self) -> tuple[int, torch.Tensor]:
        if self.num_audio_latents is not None and self._received != self.num_audio_latents:
            raise RuntimeError(f"received {self._received} audio latents, expected {self.num_audio_latents}")
        output = self._decode_until(self._received if self.num_audio_latents is None else self.num_audio_latents)
        self._latents = None
        return output
