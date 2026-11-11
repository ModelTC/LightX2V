"""Causal encoding of the current 15-second driving-audio window."""

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
        if window_start != self.window_start:
            # Shifting the left boundary changes convolution padding; cached
            # features from the previous window cannot be reused unchanged.
            self.encoder_cache.stable_features = None
            self.window_start = window_start
        waveform = self.waveform[:1] if self.mono else self.waveform
        waveform = waveform[..., window_start * self.vae.hop_length : audio_slice.stop * self.vae.hop_length]
        latents = self.vae.encode(waveform, return_cpu=False, encoder_cache=self.encoder_cache)
        if latents.shape[0] == 1:
            latents = latents.expand(2, -1, -1)
        chunk = latents[..., audio_slice.start - window_start : audio_slice.stop - window_start]
        return chunk.transpose(1, 2).reshape(-1, chunk.shape[1]).contiguous()
