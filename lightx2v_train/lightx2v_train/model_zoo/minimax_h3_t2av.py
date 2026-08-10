"""LightX2V-Train wrapper for the trainable MiniMax-H3 T2AV DiT."""

from contextlib import nullcontext

import torch
from peft import LoraConfig, inject_adapter_in_model

from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseModel
from .native.minimax_h3 import load_minimax_h3_transformer


@MODEL_REGISTER("minimax_h3_t2av")
class MiniMaxH3T2AVModel(BaseModel):
    """A normal ``nn.Module`` H3 model suitable for LoRA/full DMD training."""

    pipeline_cls = None

    def load_components(self, transformer_only=False, reference_model=None):
        del transformer_only, reference_model
        config = self.config["model"]
        self.pretrained_model_path = config["pretrained_model_name_or_path"]
        self.transformer_param_dtype = get_running_dtype(config.get("transformer_param_dtype", "bf16"))
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.video_latent_channels = int(config.get("video_latent_channels", 24))
        self.audio_latent_channels = int(config.get("audio_latent_channels", 32))
        self.vae_spatial_scale_factor = int(config.get("vae_spatial_scale_factor", 16))
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
            target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        try:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer, adapter_name="default")
        except TypeError:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer)

    def prepare_text_condition(self, condition):
        if not isinstance(condition, dict):
            raise TypeError(f"MiniMax-H3 cached condition must be a dict, got {type(condition)!r}.")
        if "prompt_embeds" not in condition or "text_token_tags" not in condition:
            raise KeyError("MiniMax-H3 condition requires prompt_embeds and text_token_tags.")
        prompt_embeds = condition["prompt_embeds"]
        text_token_tags = condition["text_token_tags"]
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
        raise RuntimeError("MiniMax-H3 DMD expects cached prompt latents. Run build_minimax_h3_prompt_latents.py first.")

    def enable_gradient_checkpointing(self):
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        else:
            self.transformer.gradient_checkpointing = True

    def fsdp2_shard_plan(self, fsdp_config):
        reshard = fsdp_config.get("reshard_after_forward", {})
        blocks = list(self.transformer.token_refiner.refiner_blocks) + list(self.transformer.transformer_blocks)
        return [
            {"modules": blocks, "reshard_after_forward": reshard.get("block_reshard", True)},
            {"module": self.transformer, "reshard_after_forward": reshard.get("root_reshard", False)},
        ]
