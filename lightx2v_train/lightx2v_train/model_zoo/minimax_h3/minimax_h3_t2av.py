"""LightX2V-Train wrapper for the trainable MiniMax-H3 T2AV DiT."""

from collections.abc import Mapping
from contextlib import nullcontext

import torch
from peft import LoraConfig, inject_adapter_in_model

from lightx2v_train.model_capabilities import ConsistencyModelCapability, DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.capability_adapters import MiniMaxH3ConsistencyModelCapability, MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.native.minimax_h3 import load_minimax_h3_transformer
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from ..base import BaseModel


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
        self.capabilities.register(
            DistributionMatchingCapability,
            MiniMaxH3DistributionMatchingCapability(
                self,
                capability_config.get("distribution_matching"),
            ),
        )
        self.capabilities.register(
            ConsistencyModelCapability,
            MiniMaxH3ConsistencyModelCapability(self),
        )

    def load_components(self, transformer_only=False, reference_model=None):
        del transformer_only, reference_model
        config = self.config["model"]
        self.pretrained_model_path = config["pretrained_model_name_or_path"]
        self.transformer_param_dtype = get_running_dtype(config.get("transformer_param_dtype", "bf16"))
        self.patch_size = tuple(int(value) for value in config.get("patch_size", (1, 2, 2)))
        if len(self.patch_size) != 3:
            raise ValueError(f"model.patch_size must contain three integers, got {self.patch_size}.")
        self.video_latent_channels = int(config.get("video_latent_channels", 24))
        self.audio_latent_channels = int(config.get("audio_latent_channels", 32))
        self.vae_spatial_scale_factor = int(config.get("vae_spatial_scale_factor", 16))
        if self.video_latent_channels <= 0 or self.audio_latent_channels <= 0 or self.vae_spatial_scale_factor <= 0:
            raise ValueError("MiniMax-H3 latent channels and VAE spatial scale must be positive.")
        self.use_autocast = bool(config.get("use_autocast", False))
        self.transformer = load_minimax_h3_transformer(
            self.pretrained_model_path,
            torch_dtype=self.transformer_param_dtype,
            local_files_only=bool(config.get("local_files_only", True)),
            attention_backend=config.get("attention_backend"),
        )
        self.transformer.to(self.device)

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
        del prompt
        raise RuntimeError("MiniMax-H3 DMD expects cached prompt latents. Run data_process/minimax_h3/build_minimax_h3_prompt_latents.py first.")

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
