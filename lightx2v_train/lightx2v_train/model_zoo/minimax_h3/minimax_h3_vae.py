"""Shared model integration for MiniMax-H3 encoder and decoder distillation."""

from contextlib import nullcontext
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

from lightx2v_train.model_capabilities import VAEDistillationCapability
from lightx2v_train.model_zoo.native.minimax_h3.video_vae import (
    imagenet_postprocess,
    imagenet_preprocess,
    normalize_video_latents,
)

from ..base import BaseModel
from .capability_adapters.minimax_h3_vae_distillation_capability import (
    MiniMaxH3VAEDistillationCapability,
)


class MiniMaxH3VAEModel(BaseModel):
    pipeline_cls = None

    @staticmethod
    def _maximum_distillation_weight(config, name):
        base = float(config.get(f"{name}_weight", 0.0))
        return max(
            [base]
            + [float(stage.get("weights", {}).get(name, base)) for stage in config.get("stages", ())]
        )

    def register_capabilities(self):
        super().register_capabilities()
        self.capabilities.register(
            VAEDistillationCapability,
            MiniMaxH3VAEDistillationCapability(
                self,
                self.config["training"].get("vae_distillation", {}),
            ),
        )

    def _load_initial_weights(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Initial VAE weights not found: {checkpoint_path}")
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            self.validate_consolidated_metadata(checkpoint.metadata() or {}, checkpoint_path)
        self.transformer.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
        logger.info("[model] initialized MiniMax-H3 VAE from {}", checkpoint_path)

    def denoiser_module(self):
        return self.transformer

    @property
    def latent_channels(self):
        return self.denoiser_module().post_quant_conv.in_channels

    def add_lora(self, rank, alpha, target_modules):
        del rank, alpha, target_modules
        raise ValueError("MiniMax-H3 VAE distillation uses training.train_type='full'; search adapters are configured separately.")

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

    def fsdp2_shard_plan(self, fsdp_config):
        reshard = fsdp_config.get("reshard_after_forward", {})
        return [
            {
                "modules": self.transformer.fsdp_modules(),
                "reshard_after_forward": reshard.get("block_reshard", True),
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard.get("root_reshard", False),
            },
        ]

    def _autocast(self, dtype):
        if self.device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    @torch.no_grad()
    def encode_video(self, video):
        if self.teacher_vae is None or self.teacher_vae.encoder is None:
            raise RuntimeError("The frozen MiniMax-H3 VAE is required to encode source videos.")
        video = video.to(device=self.device, dtype=torch.float32)
        with self._autocast(self.teacher_encoder_dtype):
            latents = self.teacher_vae.encode(imagenet_preprocess(video)).latent_dist.mode()
        return normalize_video_latents(self.teacher_vae, latents.float())

    @torch.no_grad()
    def teacher_decode_window(self, normalized_latent_window, feature_indices=()):
        if self.teacher_vae is None or self.teacher_vae.decoder is None:
            raise RuntimeError("The frozen MiniMax-H3 VAE is required for teacher decoding.")
        mean = normalized_latent_window.new_tensor(self.teacher_vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = normalized_latent_window.new_tensor(self.teacher_vae.config.latents_std).view(1, -1, 1, 1, 1)
        vae_latents = normalized_latent_window * std + mean
        captured_features = []
        handles = []
        for index in feature_indices:
            block = self.teacher_vae.decoder.transformer_blocks[index]
            handles.append(block.register_forward_hook(lambda _module, _inputs, output: captured_features.append(output)))
        try:
            with self._autocast(self.teacher_autocast_dtype):
                raw_video = self.teacher_vae.decoder(self.teacher_vae.post_quant_conv(vae_latents)).float()
        finally:
            for handle in handles:
                handle.remove()
        if not feature_indices:
            return raw_video

        batch, _, frames, height, width = normalized_latent_window.shape
        num_tokens = frames * height * width
        features = tuple(
            feature[:, :num_tokens]
            .reshape(batch, frames, height, width, feature.shape[-1])
            .permute(0, 4, 1, 2, 3)
            .float()
            for feature in captured_features
        )
        return raw_video, features

    def student_decode_window(self, normalized_latent_window, running_dtype, *, return_features=False):
        context = self._autocast(running_dtype) if self.student_autocast else nullcontext()
        with context:
            return self.denoiser_module()(normalized_latent_window, return_features=return_features)

    @staticmethod
    def postprocess_raw_video(video):
        return imagenet_postprocess(video)

    @staticmethod
    def preprocess_video(video):
        return imagenet_preprocess(video)

    @torch.no_grad()
    def decode_latents(self, normalized_latents):
        normalized_latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        self.transformer.eval()
        context = self._autocast(self.running_dtype) if self.student_autocast else nullcontext()
        with context:
            return self.transformer.decode(normalized_latents)

    @torch.no_grad()
    def reconstruct(self, video):
        return self.decode_latents(self.encode_video(video))
