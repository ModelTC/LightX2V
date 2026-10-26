"""MG4 native and tiny VAE conventions on existing Wan codec networks."""

import torch
import torch.nn.functional as F

from lightx2v.models.video_encoders.hf.tae import MemBlock
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import RMS_norm, get_wan22_vae_scale
from lightx2v.models.video_encoders.hf.wan.vae_tiny import Wan2_2_VAE_tiny


class MatrixGame4VaeRMSNorm(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.channel_first = norm.channel_first
        self.scale = norm.scale
        self.gamma = norm.gamma
        self.bias = norm.bias
        self.train(norm.training)

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias


def set_matrix_game4_vae_norm(model):
    for name, layer in tuple(model.named_modules()):
        if isinstance(layer, RMS_norm):
            model.set_submodule(name, MatrixGame4VaeRMSNorm(layer))


def _encode_streaming_features(encoder, hidden, features):
    """Tensor work shared by eager and compiled causal TAE input encoding."""
    for index, layer in enumerate(encoder):
        if isinstance(layer, MemBlock):
            shaped = hidden.unsqueeze(0)
            past = features.get(index)
            if past is None:
                past = torch.zeros_like(shaped[:, :1])
            history = torch.cat((past, shaped[:, :-1]), dim=1).reshape_as(hidden)
            features[index] = shaped[:, -1:].clone()
            hidden = layer(hidden, history)
        else:
            hidden = layer(hidden)
    return hidden


class MatrixGame4TAEStreamingEncoder:
    """Bound four-frame TAE activations while retaining causal block inputs."""

    def __init__(self, vae):
        self.model = vae.taehv
        parameter = next(self.model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.features = {}
        self.first_chunk = True

    @torch.no_grad()
    def encode(self, frames):
        expected = 1 if self.first_chunk else 4
        if frames.shape[0] != expected or frames.shape[1] != 3:
            raise ValueError("Matrix-Game-4 control chunks require one RGB frame initially, then four")
        video = frames.unsqueeze(0).to(device=self.device, dtype=self.dtype).add(1).mul(0.5)
        if self.first_chunk:
            video = torch.cat((video[:, :1].expand(-1, 3, -1, -1, -1), video), dim=1)
        hidden = F.pixel_unshuffle(video, self.model.patch_size).flatten(0, 1)
        encode = getattr(self.model, "_compiled_streaming_encoder", _encode_streaming_features)
        hidden = encode(self.model.encoder, hidden, self.features)
        self.first_chunk = False
        if hidden.shape[:2] != (1, 48):
            raise RuntimeError("Matrix-Game-4 TAE encoder must produce one 48-channel latent per chunk")
        return hidden.transpose(0, 1)

    def clear(self):
        self.features.clear()


class MatrixGame4TAEVAE(Wan2_2_VAE_tiny):
    """Use the native TAE network for both MG4 control and output pixels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.need_scaled:
            mean, std, _ = get_wan22_vae_scale(self.dtype, next(self.taehv.parameters()).device)
            self.register_buffer("decode_latents_mean", mean.view(1, 1, 48, 1, 1))
            self.register_buffer("decode_latents_std", std.view(1, 1, 48, 1, 1))

    def compile_encoder(self):
        if getattr(self.taehv, "_compiled_streaming_encoder", None) is None:
            self.taehv._compiled_streaming_encoder = torch.compile(
                _encode_streaming_features,
                fullgraph=True,
                dynamic=False,
                options={"emulate_precision_casts": True, "layout_optimization": False, "conv_1x1_as_mm": False},
            )

    @property
    def model(self):
        return self.taehv

    @torch.no_grad()
    def encode(self, video):
        if video.shape[:3] != (1, 3, 1):
            raise ValueError("Matrix-Game-4 sink encoding requires one RGB frame")
        encoder = MatrixGame4TAEStreamingEncoder(self)
        try:
            return encoder.encode(video[0].transpose(0, 1))
        finally:
            encoder.clear()

    @torch.no_grad()
    def decode(self, latents, *, memory=None, first_chunk=True):
        latents = latents.unsqueeze(0).transpose(1, 2).to(self.dtype)
        if self.need_scaled:
            # LightTAE expects raw Wan2.2 latents; Castle DiT predicts normalized z.
            latents = latents * self.decode_latents_std + self.decode_latents_mean
        images = self.taehv.decode_video(latents, parallel=False, show_progress_bar=False, memory=memory, first_chunk=first_chunk)
        return images.transpose(1, 2).mul_(2).sub_(1)

    def clear_cache(self):
        self._decode_memory = None
