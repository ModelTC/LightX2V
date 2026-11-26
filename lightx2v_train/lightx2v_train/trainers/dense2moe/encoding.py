"""Frozen native H3 AV encoders and synchronized per-stage wall timings."""

import time
from contextlib import contextmanager

import torch
from diffusers.models.attention_dispatch import attention_backend

from lightx2v_train.model_zoo.minimax_h3.minimax_h3_t2av import MiniMaxH3T2AVModel
from lightx2v_train.model_zoo.native.minimax_h3.packing import patchify_video_latents


class StageTimer:
    def __init__(self, device):
        self.device = device
        self.seconds = {}

    @contextmanager
    def measure(self, name):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        yield
        torch.cuda.synchronize(self.device)
        self.seconds[name] = time.perf_counter() - started


class OnlineAVEncoder:
    def __init__(self, model_path, device):
        self.model = MiniMaxH3T2AVModel({"model": {"running_dtype": "bf16", "pretrained_model_name_or_path": model_path}})
        self.model.device = device
        self.model.load_components(load_transformer=False, load_vae=True, load_condition_encoder=False)
        # Only encoders participate in training. Keep unused decoder weights on CPU.
        self.model.video_vae.decoder.to("cpu")
        self.model.audio_vae.decoder.to("cpu")

    @torch.no_grad()
    def encode(self, video, audio, timer):
        # Diffusers' DiT backend selection also changes its global default.
        # FP32 VAEs explicitly use SDPA; exiting restores the DiT's FA3 backend.
        with timer.measure("video_encode"), attention_backend("native"):
            video = self.model._encode_video_latents(video.float().div_(255.0))
            video = patchify_video_latents(video).unsqueeze(0)
        with timer.measure("audio_encode"), attention_backend("native"):
            audio = self.model._encode_audio_latents(audio.unsqueeze(0))
        # Native helpers use inference_mode; training inputs must be ordinary tensors.
        return {"video": video.clone(), "audio": audio.clone()}
