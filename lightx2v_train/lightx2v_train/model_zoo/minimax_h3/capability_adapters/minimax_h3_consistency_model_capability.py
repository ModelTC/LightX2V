"""Consistency training for MiniMax-H3 joint video/audio latents."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lightx2v_train.model_zoo.capability_adapters.consistency_model import (
    TimeConditionedConsistencyModelCapability,
    TimeEmbeddingAdapter,
)
from lightx2v_train.model_zoo.native.minimax_h3 import MiniMaxH3PackedSequence, build_packed_sequence


class MiniMaxH3TimeEmbeddingAdapter(TimeEmbeddingAdapter):
    """Translate noise time ``sigma`` to H3's clean-ward ``1 - sigma`` time."""

    def hook_modules(self, denoiser: nn.Module) -> tuple[nn.Module, ...]:
        return (denoiser.time_embedder,)

    def base_embedder(self, denoiser: nn.Module) -> nn.Module:
        return denoiser.time_embedder

    def embedding_dimension(self, denoiser: nn.Module) -> int:
        return int(denoiser.config.time_embed_dim)

    def encode(
        self,
        denoiser: nn.Module,
        embedder: nn.Module,
        time: Tensor,
    ) -> Tensor:
        parameter = next(embedder.parameters())
        cleanward_time = 1.0 - time.to(device=parameter.device).reshape(-1)
        projected = denoiser.time_proj(cleanward_time)
        return embedder(projected.to(device=parameter.device, dtype=parameter.dtype))


class MiniMaxH3ConsistencyModelCapability(TimeConditionedConsistencyModelCapability):
    """Expose H3's packed joint denoiser through the consistency tensor API."""

    def __init__(self, model) -> None:
        super().__init__(model, MiniMaxH3TimeEmbeddingAdapter())

    def encode_latent(self, batch):
        video_tokens, _, _, _ = self._video_tokens(batch["inputs"])
        audio_tokens = self._audio_tokens(batch["inputs"])
        return self._flatten(video_tokens, audio_tokens)

    def encode_condition(self, batch):
        positive = batch["conditioning"].get("positive")
        if positive is None:
            raise KeyError("MiniMax-H3 consistency training requires conditioning.positive with cached prompt embeddings.")
        text_condition = self.model.prepare_text_condition(positive)
        tags = text_condition["text_token_tags"]
        if not bool((tags == 1).all()):
            raise ValueError("MiniMax-H3 consistency training currently supports text-only cached conditions.")

        video_tokens, latent_frames, latent_height, latent_width = self._video_tokens(batch["inputs"])
        audio_tokens = self._audio_tokens(batch["inputs"])
        if audio_tokens.shape[1] % 2:
            raise ValueError(f"MiniMax-H3 audio token count must be even for stereo audio, got {audio_tokens.shape[1]}.")
        layout = build_packed_sequence(
            tags.detach().cpu(),
            latent_frames,
            latent_height,
            latent_width,
            audio_tokens.shape[1] // 2,
            self.model.patch_size,
        ).to(self.model.device)
        if layout.video_indices.numel() != video_tokens.shape[1]:
            raise ValueError(f"MiniMax-H3 layout has {layout.video_indices.numel()} video rows but the cache contains {video_tokens.shape[1]}.")
        if layout.audio_indices.numel() != audio_tokens.shape[1]:
            raise ValueError(f"MiniMax-H3 layout has {layout.audio_indices.numel()} audio rows but the cache contains {audio_tokens.shape[1]}.")
        return {
            **text_condition,
            "sequence_length": layout.sequence_length,
            "position_ids": layout.position_ids,
            "token_tags": layout.token_tags,
            "video_indices": layout.video_indices,
            "audio_indices": layout.audio_indices,
            "text_indices": layout.text_indices,
            "video_shape": tuple(video_tokens.shape),
            "audio_shape": tuple(audio_tokens.shape),
        }

    def sampling_latent_hw(self, batch, clean) -> tuple[int, int]:
        del clean
        _, _, latent_height, latent_width = self._video_tokens(batch["inputs"])
        return latent_height, latent_width

    def predict(self, request, path):
        with self._configured_prediction(request) as prepared_request:
            if prepared_request.model_kwargs:
                names = ", ".join(sorted(prepared_request.model_kwargs))
                raise TypeError(f"Unsupported MiniMax-H3 consistency model arguments: {names}.")
            prediction = self._predict_velocity(
                prepared_request.sample,
                prepared_request.time,
                prepared_request.condition,
            )
        return path.convert_prediction(
            prepared_request.sample,
            prediction,
            prepared_request.time,
            source_type="velocity",
            target_type=prepared_request.prediction_type,
        )

    def _predict_velocity(self, sample, sigma, condition):
        video_tokens, audio_tokens = self._unflatten(
            sample,
            condition["video_shape"],
            condition["audio_shape"],
        )
        sigma = sigma.to(device=sample.device, dtype=torch.float32).reshape(-1)
        if sigma.numel() != 1:
            raise ValueError(f"MiniMax-H3 consistency training requires one shared time value, got {tuple(sigma.shape)}.")
        layout = self._layout(condition)
        # Keep this conversion differentiable for sCM/MeanFlow forward-mode JVP.
        timesteps = 1.0 - sigma
        timestep_indices = torch.zeros(layout.sequence_length, device=self.model.device, dtype=torch.long)
        with self.model.transformer_forward_context():
            prediction = self.denoiser()(
                hidden_states=video_tokens,
                audio_hidden_states=audio_tokens,
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=timesteps.to(self.model.device),
                timestep_indices=timestep_indices.to(self.model.device),
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )
        if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
            raise TypeError("MiniMax-H3 transformer must return video and audio velocity tensors.")
        # H3 predicts x0-noise, opposite to the framework's noise-x0 RF velocity.
        return self._flatten(-prediction[0], -prediction[1])

    def _video_tokens(self, inputs):
        value = inputs.get("video_latents")
        if value is None:
            raise KeyError("MiniMax-H3 consistency training requires inputs.video_latents.")
        metadata = value if isinstance(value, dict) else {}
        tensor = self._cached_tensor(value, "video_latents")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3:
            tokens = tensor.to(device=self.model.device, dtype=self.model.running_dtype)
            latent_frames = metadata.get("latent_frames")
            latent_height = metadata.get("latent_height")
            latent_width = metadata.get("latent_width")
            if latent_frames is None or latent_height is None or latent_width is None:
                raise KeyError("Patchified MiniMax-H3 video latents require latent_frames, latent_height, and latent_width metadata.")
            geometry = int(latent_frames), int(latent_height), int(latent_width)
        elif tensor.ndim in {4, 5}:
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device=self.model.device, dtype=self.model.running_dtype)
            tokens = self._patchify_video(tensor)
            geometry = int(tensor.shape[2]), int(tensor.shape[3]), int(tensor.shape[4])
        else:
            raise ValueError(f"MiniMax-H3 video latents must be patchified [B,N,D] or raw [B,C,F,H,W], got {tuple(tensor.shape)}.")

        expected_dimension = self.model.video_latent_channels * math.prod(self.model.patch_size)
        if tokens.shape[0] != 1 or tokens.shape[-1] != expected_dimension:
            raise ValueError(f"MiniMax-H3 video tokens must have shape [1,N,{expected_dimension}], got {tuple(tokens.shape)}.")
        return tokens.contiguous(), *geometry

    def _audio_tokens(self, inputs):
        value = inputs.get("audio_latents")
        if value is None:
            raise KeyError("MiniMax-H3 consistency training requires inputs.audio_latents.")
        tensor = self._cached_tensor(value, "audio_latents")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != self.model.audio_latent_channels:
            raise ValueError(f"MiniMax-H3 audio latents must have shape [1,N,{self.model.audio_latent_channels}], got {tuple(tensor.shape)}.")
        return tensor.to(device=self.model.device, dtype=self.model.running_dtype).contiguous()

    @staticmethod
    def _cached_tensor(value, name):
        if isinstance(value, dict):
            value = value.get("tokens", value.get("latents"))
        if not torch.is_tensor(value):
            raise TypeError(f"MiniMax-H3 {name} cache must contain a tensor, got {type(value)!r}.")
        return value

    def _patchify_video(self, video):
        batch, channels, frames, height, width = video.shape
        patch_t, patch_h, patch_w = self.model.patch_size
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError(f"MiniMax-H3 raw video latent shape {tuple(video.shape)} is not divisible by patch size {self.model.patch_size}.")
        return (
            video.reshape(
                batch,
                channels,
                frames // patch_t,
                patch_t,
                height // patch_h,
                patch_h,
                width // patch_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(batch, -1, channels * patch_t * patch_h * patch_w)
        )

    @staticmethod
    def _flatten(video, audio):
        return torch.cat((video.flatten(1), audio.flatten(1)), dim=1)

    @staticmethod
    def _unflatten(flat, video_shape, audio_shape):
        video_numel = math.prod(video_shape)
        audio_numel = math.prod(audio_shape)
        if flat.shape != (1, video_numel + audio_numel):
            raise ValueError(f"Flattened MiniMax-H3 latent must have shape {(1, video_numel + audio_numel)}, got {tuple(flat.shape)}.")
        video, audio = flat.split((video_numel, audio_numel), dim=1)
        return video.reshape(video_shape), audio.reshape(audio_shape)

    @staticmethod
    def _layout(condition):
        return MiniMaxH3PackedSequence(
            sequence_length=int(condition["sequence_length"]),
            position_ids=condition["position_ids"],
            token_tags=condition["token_tags"],
            video_indices=condition["video_indices"],
            audio_indices=condition["audio_indices"],
            text_indices=condition["text_indices"],
        )
