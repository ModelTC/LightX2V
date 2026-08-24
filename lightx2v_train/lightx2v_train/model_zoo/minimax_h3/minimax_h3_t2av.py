"""LightX2V-Train wrapper for the trainable MiniMax-H3 T2AV DiT."""

from collections.abc import Mapping
from contextlib import nullcontext

import torch
from peft import LoraConfig, inject_adapter_in_model

from lightx2v_train.model_capabilities import (
    DistributionMatchingCapability,
    FlowMatchingSFTCapability,
)
from lightx2v_train.model_zoo.minimax_h3.capability_adapters import (
    MiniMaxH3DistributionMatchingCapability,
    MiniMaxH3FlowMatchingCapability,
)
from lightx2v_train.model_zoo.native.minimax_h3 import load_minimax_h3_transformer
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from ..base import BaseModel
from .condition_encoder import MiniMaxH3ConditionEncoder


@MODEL_REGISTER("minimax_h3_t2av")
class MiniMaxH3T2AVModel(BaseModel):
    """A standard trainable wrapper around Diffusers' MiniMax-H3 module."""

    pipeline_cls = None

    def register_capabilities(self):
        super().register_capabilities()
        capability_config = self.config["model"].get("capabilities", {})
        if not isinstance(capability_config, Mapping):
            raise ValueError("model.capabilities must be a mapping.")
        if "distillation" in capability_config:
            raise ValueError("model.capabilities.distillation was renamed to model.capabilities.distribution_matching.")
        unsupported_capabilities = set(capability_config) - {
            "flow_matching",
            "distribution_matching",
        }
        if unsupported_capabilities:
            names = ", ".join(sorted(unsupported_capabilities))
            raise ValueError(f"MiniMax-H3 currently supports only SFT (flow_matching) and DMD (distribution_matching); unsupported model capabilities: {names}.")
        self.capabilities.register(
            FlowMatchingSFTCapability,
            MiniMaxH3FlowMatchingCapability(
                self,
                capability_config.get("flow_matching"),
            ),
        )
        self.capabilities.register(
            DistributionMatchingCapability,
            MiniMaxH3DistributionMatchingCapability(
                self,
                capability_config.get("distribution_matching"),
            ),
        )

    def load_components(
        self,
        *,
        load_transformer,
        load_vae,
        load_condition_encoder,
    ):
        config = self.config["model"]
        self.pretrained_model_path = config["pretrained_model_name_or_path"]
        self.transformer_param_dtype = get_running_dtype(config.get("transformer_param_dtype", "bf16"))
        self.latent_dtype = get_running_dtype(config.get("latent_dtype", "fp32"))
        self.patch_size = tuple(int(value) for value in config.get("patch_size", (1, 2, 2)))
        if len(self.patch_size) != 3:
            raise ValueError(f"model.patch_size must contain three integers, got {self.patch_size}.")
        self.video_latent_channels = int(config.get("video_latent_channels", 24))
        self.audio_latent_channels = int(config.get("audio_latent_channels", 32))
        self.vae_spatial_scale_factor = int(config.get("vae_spatial_scale_factor", 16))
        self.audio_sampling_rate = int(config.get("audio_sampling_rate", 32000))
        if self.video_latent_channels <= 0 or self.audio_latent_channels <= 0 or self.vae_spatial_scale_factor <= 0 or self.audio_sampling_rate <= 0:
            raise ValueError("MiniMax-H3 latent channels, VAE spatial scale, and audio rate must be positive.")
        self.use_autocast = bool(config.get("use_autocast", False))
        self.transformer = None
        self.video_vae = None
        self.audio_vae = None
        self.condition_encoder = None
        if load_vae:
            self._load_vaes(config)
        if load_condition_encoder:
            self.condition_encoder = MiniMaxH3ConditionEncoder(
                self.pretrained_model_path,
                device=self.device,
                dtype=self.running_dtype,
                local_files_only=bool(config.get("local_files_only", True)),
                cpu_offload=bool(
                    config.get(
                        "cache_condition_encoder_cpu_offload",
                        config.get("cache_encoder_cpu_offload", False),
                    )
                ),
                attention_backend=config.get(
                    "condition_attention_backend",
                    "torch_sdpa",
                ),
            )
        if load_transformer:
            self.transformer = load_minimax_h3_transformer(
                self.pretrained_model_path,
                torch_dtype=self.transformer_param_dtype,
                local_files_only=bool(config.get("local_files_only", True)),
                attention_backend=config.get("attention_backend"),
            )
            self.transformer.to(self.device)

    def _load_vaes(self, config):
        try:
            from lightx2v.models.audio_encoders.hf.minimax_h3 import MiniMaxH3AudioVAE
            from lightx2v.models.video_encoders.hf.minimax_h3 import MiniMaxH3VideoVAE
        except ImportError as error:
            raise ImportError("MiniMax-H3 cache construction requires LightX2V's native video and audio VAEs.") from error

        cpu_offload = bool(config.get("cache_encoder_cpu_offload", False))
        self.video_vae = MiniMaxH3VideoVAE.from_pretrained(
            self.pretrained_model_path,
            device=self.device,
            cpu_offload=cpu_offload,
            sensitive_layer_dtype=torch.float32,
            attn_type=config.get("vae_attention_backend", "torch_sdpa"),
        )
        self.audio_vae = MiniMaxH3AudioVAE.from_pretrained(
            self.pretrained_model_path,
            device=self.device,
            cpu_offload=cpu_offload,
        )
        if int(self.audio_vae.sampling_rate) != self.audio_sampling_rate:
            raise ValueError(f"MiniMax-H3 audio_sampling_rate does not match the audio VAE: config={self.audio_sampling_rate}, checkpoint={self.audio_vae.sampling_rate}.")

    def reuse_frozen_components_from(self, source):
        super().reuse_frozen_components_from(source)
        self.video_vae = source.video_vae
        self.audio_vae = source.audio_vae
        self.condition_encoder = source.condition_encoder

    def denoiser_module(self):
        return self.transformer

    def transformer_forward_context(self):
        if self.use_autocast and self.device.type == "cuda" and self.running_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast("cuda", dtype=self.running_dtype)
        return nullcontext()

    def add_lora(self, rank, alpha, target_modules):
        if not target_modules:
            target_modules = MiniMaxH3DistributionMatchingCapability._DEFAULT_LORA_TARGETS
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        try:
            self.transformer = inject_adapter_in_model(
                lora_config,
                self.transformer,
                adapter_name="default",
            )
        except TypeError:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer)

    def prepare_text_condition(self, condition):
        if not isinstance(condition, dict):
            raise TypeError(f"MiniMax-H3 cached condition must be a dict, got {type(condition)!r}.")
        missing = {"prompt_embeds", "text_token_tags"} - condition.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"MiniMax-H3 cached condition is missing: {names}.")

        prompt_embeds = condition["prompt_embeds"]
        text_token_tags = condition["text_token_tags"]
        if not torch.is_tensor(prompt_embeds) or not torch.is_tensor(text_token_tags):
            raise TypeError("MiniMax-H3 prompt_embeds and text_token_tags must be tensors.")
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
            raise ValueError(f"MiniMax-H3 prompt_embeds must have shape [1, tokens, dim], got {tuple(prompt_embeds.shape)}.")
        if text_token_tags.ndim == 2:
            if text_token_tags.shape[0] != 1:
                raise ValueError("MiniMax-H3 currently requires data.train.batch_size=1.")
            text_token_tags = text_token_tags[0]
        if text_token_tags.ndim != 1 or text_token_tags.shape[0] != prompt_embeds.shape[1]:
            raise ValueError(f"MiniMax-H3 text_token_tags must contain one tag per prompt embedding row; got {tuple(text_token_tags.shape)} for {prompt_embeds.shape[1]} rows.")
        return {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=self.running_dtype),
            "text_token_tags": text_token_tags.to(self.device, dtype=torch.long),
        }

    def encode_prompt_condition(self, prompt):
        if self.condition_encoder is None:
            raise RuntimeError("MiniMax-H3 condition encoder is not loaded. Use a training cache or load condition components.")
        return self.prepare_text_condition(self.condition_encoder.encode(prompt))

    def encode_condition(self, sample):
        conditioning = sample.get("conditioning", {})
        cached = conditioning.get(conditioning.get("active", "positive"))
        if cached is not None:
            return self.prepare_text_condition(cached)
        return self.encode_prompt_condition(sample["conditioning"]["prompt"])

    def encode_to_cache_latents(self, sample):
        inputs = sample.get("inputs", {})
        video_latents = inputs.get("video_latents")
        audio_latents = inputs.get("audio_latents")
        if video_latents is not None and audio_latents is not None:
            return {
                "video_latents": video_latents,
                "audio_latents": audio_latents,
            }
        if self.video_vae is None or self.audio_vae is None:
            raise RuntimeError("MiniMax-H3 VAEs are not loaded. Use cached latents or load VAE components.")

        video = inputs.get("video")
        audio = inputs.get("audio")
        if not torch.is_tensor(video) or not torch.is_tensor(audio):
            raise KeyError("MiniMax-H3 cache encoding requires inputs.video and inputs.audio.")
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or video.shape[:2] != (1, 3):
            raise ValueError(f"MiniMax-H3 video must be [1,3,F,H,W], got {tuple(video.shape)}.")
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        if audio.ndim != 3 or audio.shape[:2] != (1, 2):
            raise ValueError(f"MiniMax-H3 audio must be [1,2,samples], got {tuple(audio.shape)}.")

        video_latents = self.video_vae.encode_condition(
            video,
            video=True,
            return_cpu=False,
            sample_posterior=False,
        )
        stereo_waveform = audio[0].to(device=self.device, dtype=torch.float32)
        audio_latents = self.audio_vae.encode(stereo_waveform, return_cpu=False)
        audio_latents = audio_latents.transpose(1, 2).reshape(
            1,
            -1,
            audio_latents.shape[1],
        )
        return {
            "video_latents": video_latents,
            "audio_latents": audio_latents,
        }

    def enable_gradient_checkpointing(self):
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        else:
            self.transformer.gradient_checkpointing = True

    def fsdp2_shard_plan(self, fsdp_config):
        reshard = fsdp_config.get("reshard_after_forward", {})
        blocks = list(self.transformer.token_refiner.refiner_blocks) + list(self.transformer.transformer_blocks)
        return [
            {
                "modules": blocks,
                "reshard_after_forward": reshard.get("block_reshard", True),
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard.get("root_reshard", False),
            },
        ]
