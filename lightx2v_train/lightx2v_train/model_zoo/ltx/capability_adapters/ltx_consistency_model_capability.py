"""Consistency training for LTX's joint video/audio latent state."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from lightx2v_train.model_zoo.capability_adapters.consistency_model import (
    TimeConditionedConsistencyModelCapability,
    TimeEmbeddingAdapter,
)
from lightx2v_train.model_zoo.native.ltx2 import Modality

from .common import LTXLatentCodec


class LTXTimeEmbeddingAdapter(TimeEmbeddingAdapter):
    """Condition both LTX modalities on a shared MeanFlow endpoint."""

    def hook_modules(self, denoiser: nn.Module) -> tuple[nn.Module, ...]:
        return denoiser.adaln_single, denoiser.audio_adaln_single

    def base_embedder(self, denoiser: nn.Module) -> nn.Module:
        return denoiser.adaln_single

    def build_endpoint_embedder(self, denoiser: nn.Module) -> nn.Module:
        return nn.ModuleDict(
            {
                "video": copy.deepcopy(denoiser.adaln_single),
                "audio": copy.deepcopy(denoiser.audio_adaln_single),
            }
        )

    def embedding_dimension(self, denoiser: nn.Module) -> int:
        return int(denoiser.inner_dim)

    def encode(
        self,
        denoiser: nn.Module,
        embedder: nn.Module,
        time: Tensor,
    ) -> Tensor:
        parameter = next(embedder.parameters())
        scaled_time = time.to(device=parameter.device) * float(denoiser.timestep_scale_multiplier)
        _, embedding = embedder(
            scaled_time.flatten(),
            hidden_dtype=parameter.dtype,
        )
        return embedding

    def add_endpoint_embedding(
        self,
        denoiser: nn.Module,
        endpoint_embedder: nn.ModuleDict,
        endpoint_time: Tensor,
        hook_module: nn.Module,
        output,
    ):
        if not isinstance(output, tuple) or len(output) != 2 or not all(torch.is_tensor(value) for value in output):
            raise TypeError("LTX AdaLN endpoint conditioning expects a pair of tensor outputs.")
        if hook_module is denoiser.adaln_single:
            branch = endpoint_embedder["video"]
        elif hook_module is denoiser.audio_adaln_single:
            branch = endpoint_embedder["audio"]
        else:
            raise RuntimeError("Received an endpoint hook from an unknown LTX AdaLN module.")

        parameter = next(branch.parameters())
        scaled_time = endpoint_time.to(device=parameter.device) * float(denoiser.timestep_scale_multiplier)
        endpoint_outputs = branch(
            scaled_time.flatten(),
            hidden_dtype=parameter.dtype,
        )
        return tuple(self._add_broadcast(value, endpoint) for value, endpoint in zip(output, endpoint_outputs))

    @staticmethod
    def _add_broadcast(value: Tensor, endpoint: Tensor) -> Tensor:
        while endpoint.ndim < value.ndim:
            endpoint = endpoint.unsqueeze(-2)
        return value + endpoint.to(device=value.device, dtype=value.dtype)


class LTXConsistencyModelCapability(TimeConditionedConsistencyModelCapability):
    """Expose joint LTX audio/video denoising through the tensor CM boundary."""

    def __init__(self, model) -> None:
        super().__init__(model, LTXTimeEmbeddingAdapter())
        config = model.config["training"].get("ltx2", {})
        self.latent_codec = LTXLatentCodec(
            model,
            default_fps=float(config.get("default_fps", 24.0)),
        )

    def encode_latent(self, batch):
        _, _, _, _, video_tokens, audio_tokens = self.latent_codec.prepare(
            batch["inputs"],
            self.model.running_dtype,
        )
        return self.latent_codec.flatten(video_tokens, audio_tokens)

    def encode_condition(self, batch):
        conditioning = batch["conditioning"]
        positive = conditioning.get("positive")
        if positive is None:
            video_context, audio_context, context_mask = self.model.encode_prompt_condition(conditioning.get("prompt", ""))
        else:
            video_context, audio_context, context_mask = self.model.prepare_text_condition(positive)

        video_data, _, video_latents, _, video_tokens, audio_tokens = self.latent_codec.prepare(
            batch["inputs"],
            self.model.running_dtype,
        )
        video_positions, audio_positions = self.latent_codec.positions(
            video_data,
            video_latents,
            audio_tokens,
        )
        return {
            "video_context": video_context,
            "audio_context": audio_context,
            "context_mask": context_mask,
            "video_positions": video_positions,
            "audio_positions": audio_positions,
            "video_shape": tuple(video_tokens.shape),
            "audio_shape": tuple(audio_tokens.shape),
        }

    def sampling_latent_hw(self, batch, clean) -> tuple[int, int]:
        del clean
        _, _, video_latents, _, _, _ = self.latent_codec.prepare(
            batch["inputs"],
            self.model.running_dtype,
        )
        return int(video_latents.shape[-2]), int(video_latents.shape[-1])

    def predict(self, request, path):
        with self._configured_prediction(request) as prepared_request:
            if prepared_request.model_kwargs:
                names = ", ".join(sorted(prepared_request.model_kwargs))
                raise TypeError(f"Unsupported LTX consistency model arguments: {names}.")
            prediction = self._predict_velocity(
                prepared_request.sample,
                prepared_request.time,
                prepared_request.condition,
            )
        return path.convert_prediction(
            prepared_request.sample,
            prediction,
            prepared_request.time,
            source_type=self.model.denoiser_prediction_type(),
            target_type=prepared_request.prediction_type,
        )

    def _predict_velocity(self, sample, time, condition):
        video_tokens, audio_tokens = self.latent_codec.unflatten(
            sample,
            condition["video_shape"],
            condition["audio_shape"],
        )
        sigma = time.to(device=sample.device, dtype=sample.dtype).reshape(-1)
        if sigma.numel() != 1:
            raise ValueError(f"LTX consistency training requires one shared time value, got {tuple(time.shape)}.")
        video_timesteps = sigma.view(1, 1).expand(1, video_tokens.shape[1])
        audio_timesteps = sigma.view(1, 1).expand(1, audio_tokens.shape[1])

        video = Modality(
            enabled=True,
            latent=video_tokens,
            sigma=sigma,
            timesteps=video_timesteps,
            positions=condition["video_positions"],
            context=condition["video_context"],
            context_mask=condition["context_mask"],
        )
        audio = Modality(
            enabled=True,
            latent=audio_tokens,
            sigma=sigma,
            timesteps=audio_timesteps,
            positions=condition["audio_positions"],
            context=condition["audio_context"],
            context_mask=condition["context_mask"],
        )
        with self.model.transformer_forward_context():
            video_prediction, audio_prediction = self.denoiser()(
                video=video,
                audio=audio,
                perturbations=None,
            )
        video_prediction = video_prediction[:, -video_tokens.shape[1] :]
        audio_prediction = audio_prediction[:, -audio_tokens.shape[1] :]
        return self.latent_codec.flatten(video_prediction, audio_prediction)
