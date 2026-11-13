"""Causal encoding of bounded 15-second offline or live driving-audio windows."""

import math

import torch
import torch.nn.functional as F
import torchaudio

from lightx2v.models.audio_encoders.hf.minimax_h3.audio_vae import MiniMaxH3AudioEncoderPrefixCache
from lightx2v.models.networks.minimax_h3.packing import audio_latent_num_frames


class MiniMaxH3DrivingAudio:
    def __init__(self, vae, path, *, num_frames, mono=False, loop=False):
        self.vae = vae
        self.mono = mono
        waveform, sample_rate = torchaudio.load(path)
        if not waveform.numel() or not torch.isfinite(waveform).all():
            raise ValueError("RefA2V driving audio must contain finite samples")
        if mono:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.expand(2, -1) if waveform.shape[0] == 1 else waveform[:2]
        if sample_rate != vae.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae.sampling_rate)
        samples = audio_latent_num_frames(num_frames) * vae.hop_length
        if loop and waveform.shape[-1] < samples:
            waveform = waveform.repeat(1, math.ceil(samples / waveform.shape[-1]))
        self.waveform = F.pad(waveform[..., :samples], (0, max(0, samples - waveform.shape[-1]))).contiguous()
        self.encoder_cache = MiniMaxH3AudioEncoderPrefixCache(vae.encoder)
        self.window_start = 0

    def encode(self, audio_slice):
        window_latents = 15 * self.vae.sampling_rate // self.vae.hop_length
        window_start = max(0, audio_slice.stop - window_latents)
        waveform = self.waveform[..., window_start * self.vae.hop_length : audio_slice.stop * self.vae.hop_length]
        return self._encode_window(waveform, audio_slice, window_start)

    def _encode_window(self, waveform, audio_slice, window_start):
        if window_start != self.window_start:
            # Shifting the left boundary changes convolution padding; cached
            # features from the previous window cannot be reused unchanged.
            self.encoder_cache.stable_features = None
            self.window_start = window_start
        waveform = waveform[:1] if self.mono else waveform
        latents = self.vae.encode(waveform, return_cpu=False, encoder_cache=self.encoder_cache)
        if latents.shape[0] == 1:
            latents = latents.expand(2, -1, -1)
        chunk = latents[..., audio_slice.start - window_start : audio_slice.stop - window_start]
        return chunk.transpose(1, 2).reshape(-1, chunk.shape[1]).contiguous()


class MiniMaxH3StreamingDrivingAudio(MiniMaxH3DrivingAudio):
    """Append one live chunk without losing encoder context or retaining the session."""

    def __init__(self, vae, *, mono=False):
        self.vae = vae
        self.mono = mono
        self.waveform = torch.empty((1 if mono else 2, 0), dtype=torch.float32)
        self.encoder_cache = MiniMaxH3AudioEncoderPrefixCache(vae.encoder)
        self.window_start = self.received_latents = 0

    def encode(self, audio_slice, waveform, *, sample_rate=None):
        if audio_slice.start != self.received_latents or audio_slice.stop <= audio_slice.start:
            raise ValueError("Live RefA2V audio slices must be contiguous and nonempty")
        waveform = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or not waveform.numel() or not torch.isfinite(waveform).all():
            raise ValueError("Live RefA2V driving audio must contain finite mono/stereo samples")
        waveform = waveform.mean(dim=0, keepdim=True) if self.mono else waveform[:2]
        if not self.mono and waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        if sample_rate is not None and sample_rate != self.vae.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.vae.sampling_rate)
        samples = (audio_slice.stop - audio_slice.start) * self.vae.hop_length
        waveform = F.pad(waveform[..., :samples], (0, max(0, samples - waveform.shape[-1])))
        window_latents = 15 * self.vae.sampling_rate // self.vae.hop_length
        window_start = max(0, audio_slice.stop - window_latents)
        drop = (window_start - self.window_start) * self.vae.hop_length
        self.waveform = torch.cat((self.waveform, waveform), dim=-1)[..., drop:].contiguous()
        self.received_latents = audio_slice.stop
        return self._encode_window(self.waveform, audio_slice, window_start)
