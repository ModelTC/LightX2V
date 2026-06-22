import math
import os
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from loguru import logger
from peft import LoraConfig, inject_adapter_in_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseModel
from .native.wan.modules.causal_model import CausalWanModel
from .native.wan.modules.model import WanModel
from .native.wan.modules.t5 import T5EncoderModel
from .native.wan.modules.vae import WanVAE


@dataclass
class WanT2VDenoiserInput:
    hidden_states: torch.Tensor


@MODEL_REGISTER("wan_t2v")
class WanT2VModel(BaseModel):
    pipeline_cls = None

    def load_components(self, transformer_only=False, reference_model=None):
        model_config = self.config["model"]
        model_path = model_config["pretrained_model_name_or_path"]

        self.load_vae = model_config.get("load_vae", True)
        self.load_text_encoder = model_config.get("load_text_encoder", True)
        self.load_transformer = model_config.get("load_transformer", True)
        teacher_forcing_config = self.config.get("training", {}).get("teacher_forcing", {})
        self.use_causal_transformer = model_config.get("causal", False) or teacher_forcing_config.get("enabled", False)
        self.sample_posterior = model_config.get("sample_posterior", True)
        self.num_train_timesteps = self.config.get("scheduler", {}).get("num_train_timesteps", 1000)
        self.max_sequence_length = model_config.get("max_sequence_length", 512)
        default_param_dtype = "fp32" if self.config.get("training", {}).get("train_type") == "full" else model_config.get("running_dtype", "bf16")
        self.transformer_param_dtype = get_running_dtype(model_config.get("transformer_param_dtype", default_param_dtype))
        self.vae_dtype = get_running_dtype(model_config.get("vae_dtype", "fp32"))
        self.t5_dtype = get_running_dtype(model_config.get("t5_dtype", "bf16"))
        self.t5_cpu = model_config.get("t5_cpu", False)
        self.vae_stride = tuple(model_config.get("vae_stride", (4, 8, 8)))
        self.patch_size = tuple(model_config.get("patch_size", (1, 2, 2)))
        self.sp_size = int(model_config.get("sequence_parallel_size", 1))
        self.num_frame_per_chunk = int(teacher_forcing_config.get("num_frame_per_chunk", model_config.get("num_frame_per_chunk", 1)))
        self.local_attn_size = int(model_config.get("local_attn_size", teacher_forcing_config.get("local_attn_size", -1)))
        self.sink_size = int(model_config.get("sink_size", teacher_forcing_config.get("sink_size", 0)))
        self.independent_first_frame = bool(model_config.get("independent_first_frame", teacher_forcing_config.get("independent_first_frame", False)))
        self.text_encoder = None
        self.text_pipeline = None

        if transformer_only:
            if reference_model is not None:
                self.vae = reference_model.vae
                self.text_encoder = reference_model.text_encoder
                self.text_pipeline = reference_model.text_pipeline
                self.vae_stride = reference_model.vae_stride
                self.patch_size = reference_model.patch_size
                self.max_sequence_length = reference_model.max_sequence_length
                self.vae_scale_factor_temporal = reference_model.vae_scale_factor_temporal
                self.vae_scale_factor_spatial = reference_model.vae_scale_factor_spatial
            self.transformer = self._load_transformer(model_path)
            self._configure_transformer()
            self.vae_scale_factor_temporal = self.vae_stride[0]
            self.vae_scale_factor_spatial = self.vae_stride[1]
            return

        if self.load_transformer:
            self.transformer = self._load_transformer(model_path)
            self._configure_transformer()
        else:
            self.transformer = None

        if self.load_vae:
            vae_checkpoint = os.path.join(model_path, "Wan2.1_VAE.pth")
            self.vae = WanVAE(vae_pth=vae_checkpoint, dtype=self.vae_dtype, device=self.device)
            self.vae.model.requires_grad_(False)

        if self.load_text_encoder:
            t5_checkpoint = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
            t5_tokenizer = os.path.join(model_path, "google/umt5-xxl")
            self.text_encoder = T5EncoderModel(
                text_len=self.max_sequence_length,
                dtype=self.t5_dtype,
                device=torch.device("cpu"),
                checkpoint_path=t5_checkpoint,
                tokenizer_path=t5_tokenizer,
            )
            self.text_encoder.model.requires_grad_(False)
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)

        self.vae_scale_factor_temporal = self.vae_stride[0]
        self.vae_scale_factor_spatial = self.vae_stride[1]

    def _load_transformer(self, model_path):
        if self.use_causal_transformer:
            transformer = CausalWanModel.from_pretrained(
                model_path,
                torch_dtype=self.transformer_param_dtype,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
            )
        else:
            transformer = WanModel.from_pretrained(model_path, torch_dtype=self.transformer_param_dtype)
        return transformer.to(self.device, dtype=self.transformer_param_dtype)

    def _configure_transformer(self):
        self.patch_size = tuple(self.transformer.patch_size)
        self.max_sequence_length = int(getattr(self.transformer, "text_len", self.max_sequence_length))
        if isinstance(self.transformer, CausalWanModel):
            self.transformer.num_frame_per_block = self.num_frame_per_chunk
            self.transformer.independent_first_frame = self.independent_first_frame

    def denoiser_module(self):
        if self.transformer is None:
            raise RuntimeError("Wan transformer is not loaded. Set model.load_transformer=True for training.")
        return self.transformer

    def transformer_forward_context(self):
        if self.device.type == "cuda" and self.running_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.running_dtype)
        return nullcontext()

    def add_lora(self, rank, alpha, target_modules):
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self._inject_lora(lora_config)

    def _inject_lora(self, lora_config, adapter_name="default"):
        try:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer, adapter_name=adapter_name)
        except TypeError:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer)

    def _lora_config_for_infer(self):
        lora_config = dict(self.config.get("training", {}).get("lora", {}))
        lora_config.update(self.config.get("inference", {}).get("lora_config", {}))
        rank = lora_config.get("rank", 64)
        alpha = lora_config.get("alpha", rank)
        target_modules = lora_config.get("target_modules", ["q", "k", "v", "o", "ffn.0", "ffn.2"])
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        if adapter_name is None:
            adapter_name = "default"
        if not hasattr(self.denoiser_module(), "peft_config"):
            self._inject_lora(self._lora_config_for_infer(), adapter_name=adapter_name)

        weight_path = lora_path
        if os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, "pytorch_lora_weights.safetensors")
        raw = load_file(weight_path)
        peft_state_dict = {}
        for key, value in raw.items():
            new_key = key.removeprefix("transformer.")
            new_key = new_key.replace(".lora.down.weight", ".lora_A.weight")
            new_key = new_key.replace(".lora.up.weight", ".lora_B.weight")
            peft_state_dict[new_key] = value

        incompatible = set_peft_model_state_dict(self.denoiser_module(), peft_state_dict)
        if incompatible and incompatible.unexpected_keys:
            logger.warning("Unexpected keys when loading Wan LoRA: {}", incompatible.unexpected_keys)
        self._infer_lora_adapter_name = adapter_name

    def unload_lora_for_infer(self):
        adapter_name = getattr(self, "_infer_lora_adapter_name", None)
        if adapter_name is None:
            return

        denoiser = self.denoiser_module()
        if hasattr(denoiser, "delete_adapter"):
            denoiser.delete_adapter(adapter_name)
        elif hasattr(denoiser, "delete_adapters"):
            denoiser.delete_adapters(adapter_name)
        else:
            self.transformer = self._load_transformer(self.config["model"]["pretrained_model_name_or_path"])
        self._infer_lora_adapter_name = None

    def fsdp2_shard_plan(self, fsdp_config):
        reshard_config = fsdp_config.get(
            "reshard_after_forward",
            {
                "root_reshard": False,
                "block_reshard": True,
            },
        )
        return [
            {
                "modules": self.transformer.blocks,
                "reshard_after_forward": reshard_config.get("block_reshard", True),
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard_config.get("root_reshard", False),
            },
        ]

    def encode_to_latent(self, sample):
        if "latent" in sample:
            latent = sample["latent"].to(device=self.device, dtype=self.running_dtype)
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            return latent

        if self.vae is None:
            raise RuntimeError("Wan VAE is not loaded. Use cached latents or set model.load_vae=True.")

        video = sample["video"].to(device=self.device, dtype=self.vae_dtype)
        latent = torch.stack(self.vae.encode(self._batch_to_list(video)), dim=0)
        return latent.to(dtype=self.running_dtype)

    def encode_condition(self, sample):
        if "prompt_embed" in sample:
            prompt_embed = sample["prompt_embed"].to(device=self.device, dtype=self.running_dtype)
            if prompt_embed.ndim == 2:
                prompt_embed = prompt_embed.unsqueeze(0)
            return {"prompt_embed": prompt_embed}

        if self.text_encoder is None:
            raise RuntimeError("Wan text encoder is not loaded. Use cached prompt embeds or set model.load_text_encoder=True.")

        prompt = sample["prompt"]
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        text_device = torch.device("cpu") if self.t5_cpu else self.device
        contexts = self.text_encoder(prompts, text_device)
        prompt_embed = self._pad_contexts(contexts, device=self.device, dtype=self.running_dtype)
        return {"prompt_embed": prompt_embed}

    def encode_prompt_condition(self, prompt):
        return self.encode_condition({"prompt": prompt})

    def prepare_denoiser_input(self, noisy_latent):
        return WanT2VDenoiserInput(hidden_states=noisy_latent)

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        timestep = timestep_or_sigma.float() * self.num_train_timesteps
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(device=self.device)

        hidden_states = denoiser_input.hidden_states.to(device=self.device, dtype=self.running_dtype)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.unsqueeze(0)
        seq_len = self._sequence_length(hidden_states)

        if isinstance(self.transformer, CausalWanModel):
            timestep = self._causal_timestep(timestep, hidden_states)
            context = self._condition_to_context_tensor(condition, batch_size=hidden_states.shape[0])
            self._prepare_causal_block_mask(hidden_states, teacher_forcing=False)
            with self.transformer_forward_context():
                return self.transformer(
                    hidden_states,
                    t=timestep,
                    context=context,
                    seq_len=seq_len,
                )

        latent_list = self._batch_to_list(hidden_states)
        context = self._condition_to_context_list(condition, batch_size=len(latent_list))
        with self.transformer_forward_context():
            return self.transformer(
                latent_list,
                t=timestep,
                context=context,
                seq_len=seq_len,
            )

    def denoise_teacher_forcing(self, noisy_latent, timestep_or_sigma, condition, clean_latent, aug_timestep_or_sigma=None):
        if not isinstance(self.transformer, CausalWanModel):
            raise RuntimeError("Wan teacher forcing requires model.causal=True or training.teacher_forcing.enabled=True.")

        timestep = timestep_or_sigma.float() * self.num_train_timesteps
        timestep = timestep.to(device=self.device)

        aug_t = None
        if aug_timestep_or_sigma is not None:
            aug_t = aug_timestep_or_sigma.float() * self.num_train_timesteps
            aug_t = aug_t.to(device=self.device)

        noisy_latent = noisy_latent.to(device=self.device, dtype=self.running_dtype)
        clean_latent = clean_latent.to(device=self.device, dtype=self.running_dtype)
        if noisy_latent.ndim == 4:
            noisy_latent = noisy_latent.unsqueeze(0)
        if clean_latent.ndim == 4:
            clean_latent = clean_latent.unsqueeze(0)
        timestep = self._causal_timestep(timestep, noisy_latent)
        if aug_t is not None:
            aug_t = self._causal_timestep(aug_t, noisy_latent)
        context = self._condition_to_context_tensor(condition, batch_size=noisy_latent.shape[0])
        seq_len = self._sequence_length(noisy_latent)
        self._prepare_causal_block_mask(noisy_latent, teacher_forcing=True)

        with self.transformer_forward_context():
            return self.transformer(
                noisy_latent,
                t=timestep,
                context=context,
                seq_len=seq_len,
                clean_x=clean_latent,
                aug_t=aug_t,
            )

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return prediction

    def prepare_infer_latents(self, height, width, generator=None):
        num_frames = self.config.get("inference", {}).get("num_frames", 81)
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        in_channels = self._latent_channels()
        shape = (
            1,
            in_channels,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def decode_latent(self, latent):
        if self.vae is None:
            raise RuntimeError("Wan VAE is not loaded. Set model.load_vae=True for decoding.")

        latent = latent.to(device=self.device, dtype=self.vae_dtype)
        videos = torch.stack(self.vae.decode(self._batch_to_list(latent)), dim=0)
        return self._postprocess_videos(videos)

    def assemble_pipeline(self, scheduler=None):
        raise NotImplementedError("Native Wan T2V uses wan_t2v_infer instead of assembling a diffusers pipeline.")

    def get_pipeline_infer_kwargs(self, infer_config):
        return {
            "height": infer_config.get("height", infer_config.get("default_height", 480)),
            "width": infer_config.get("width", infer_config.get("default_width", 832)),
            "num_frames": infer_config.get("num_frames", 81),
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "guidance_scale": infer_config.get("cfg_guidance_scale", 5.0),
        }

    def _batch_to_list(self, tensor):
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        return [item for item in tensor.unbind(0)]

    def _pad_contexts(self, contexts, device, dtype):
        padded = []
        for context in contexts:
            context = context[: self.max_sequence_length].to(device=device, dtype=dtype)
            if context.shape[0] < self.max_sequence_length:
                pad = context.new_zeros(self.max_sequence_length - context.shape[0], context.shape[1])
                context = torch.cat([context, pad], dim=0)
            padded.append(context)
        return torch.stack(padded, dim=0)

    def _condition_to_context_list(self, condition, batch_size):
        return [item for item in self._condition_to_context_tensor(condition, batch_size=batch_size).unbind(0)]

    def _condition_to_context_tensor(self, condition, batch_size):
        prompt_embed = condition["prompt_embed"].to(device=self.device, dtype=self.running_dtype)
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)
        prompt_embed = prompt_embed[:, : self.max_sequence_length]
        if prompt_embed.shape[0] == 1 and batch_size > 1:
            prompt_embed = prompt_embed.expand(batch_size, -1, -1)
        elif prompt_embed.shape[0] != batch_size:
            raise ValueError(f"Prompt embed batch size {prompt_embed.shape[0]} does not match latent batch size {batch_size}.")
        return prompt_embed

    def _causal_timestep(self, timestep, latent):
        batch_size, num_frames = latent.shape[0], latent.shape[2]
        if timestep.ndim == 0:
            timestep = timestep.reshape(1, 1).expand(batch_size, num_frames)
        elif timestep.ndim == 1:
            if timestep.shape[0] == batch_size:
                timestep = timestep[:, None].expand(batch_size, num_frames)
            elif timestep.shape[0] == num_frames and batch_size == 1:
                timestep = timestep[None, :]
            elif timestep.shape[0] == 1:
                timestep = timestep.reshape(1, 1).expand(batch_size, num_frames)
            else:
                raise ValueError(f"Causal Wan timestep shape {tuple(timestep.shape)} does not match latent shape {tuple(latent.shape)}.")
        elif timestep.ndim == 2:
            if timestep.shape[0] == 1 and batch_size > 1:
                timestep = timestep.expand(batch_size, -1)
            if timestep.shape[1] == 1 and num_frames > 1:
                timestep = timestep.expand(-1, num_frames)
            if timestep.shape != (batch_size, num_frames):
                raise ValueError(f"Causal Wan timestep shape {tuple(timestep.shape)} does not match latent shape {tuple(latent.shape)}.")
        else:
            raise ValueError(f"Causal Wan timestep must be scalar, [B], or [B, F], got shape {tuple(timestep.shape)}.")
        return timestep

    def _prepare_causal_block_mask(self, latent, teacher_forcing):
        key = (
            bool(teacher_forcing),
            int(latent.shape[2]),
            int(latent.shape[-2]),
            int(latent.shape[-1]),
            tuple(self.patch_size),
            int(self.num_frame_per_chunk),
            int(self.local_attn_size),
            bool(self.independent_first_frame),
        )
        if getattr(self.transformer, "_lightx2v_block_mask_key", None) != key:
            self.transformer.block_mask = None
            self.transformer._lightx2v_block_mask_key = key

    def _sequence_length(self, latent):
        latent_frames, latent_height, latent_width = latent.shape[-3:]
        patch_t, patch_h, patch_w = self.patch_size
        seq_len = (latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
        return math.ceil(seq_len / self.sp_size) * self.sp_size

    def _latent_channels(self):
        if self.transformer is not None:
            return int(getattr(self.transformer, "in_dim", getattr(self.transformer.config, "in_dim", 16)))
        return int(self.config["model"].get("latent_channels", 16))

    def _postprocess_videos(self, videos):
        videos = videos.detach().float().cpu().clamp(-1, 1)
        videos = ((videos + 1.0) * 127.5).round().to(torch.uint8)
        result = []
        for video in videos:
            frames = video.permute(1, 2, 3, 0).numpy()
            result.append([Image.fromarray(np.ascontiguousarray(frame)) for frame in frames])
        return result
