"""LightX2V-Train wrapper for the trainable MiniMax-H3 T2AV DiT."""

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from math import prod
from pathlib import Path

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
from lightx2v_train.model_zoo.native.minimax_h3 import (
    init_empty_minimax_h3_transformer,
    load_minimax_h3_transformer,
    resolve_transformer_dir,
    stream_load_minimax_h3_transformer,
)
from lightx2v_train.runtime.fsdp import fsdp2_enabled
from lightx2v_train.runtime.sequence_parallel import sync_sequence_parallel_parameters
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from ..base import BaseModel
from .condition_encoder import MiniMaxH3ConditionEncoder


@MODEL_REGISTER("minimax_h3_t2av")
class MiniMaxH3T2AVModel(BaseModel):
    """A standard trainable wrapper around Diffusers' MiniMax-H3 module."""

    pipeline_cls = None
    transformer_component = "transformer"

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
        fsdp_config = self.config.get("distributed", {}).get("fsdp2", {})
        self._stream_load_pending = bool(fsdp_config.get("stream_load_pretrained", False))
        self._stream_load_lora_seed = int(config.get("lora_init_seed", 0))
        self.cache_encoder_cpu_offload = bool(config.get("cache_encoder_cpu_offload", False))
        self.transformer = None
        self.video_vae = None
        self.audio_vae = None
        self.condition_encoder = None
        self._managed_cache_encoders = set()
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
                text_encoder_subfolder=config.get("text_encoder_subfolder", "text_encoder"),
                tokenizer_subfolder=config.get("tokenizer_subfolder", "tokenizer"),
                processor_subfolder=config.get("processor_subfolder", "processor"),
                text_encoder_layer=int(config.get("text_encoder_layer", 50)),
                text_tag=int(config.get("text_tag", 1)),
            )
        if load_transformer:
            loader_kwargs = {
                "component_name": self.transformer_component,
                "torch_dtype": self.transformer_param_dtype,
                "local_files_only": bool(config.get("local_files_only", True)),
                "attention_backend": config.get("attention_backend"),
            }
            if self._stream_load_pending:
                if not fsdp2_enabled(self.config):
                    raise RuntimeError("H3 streamed checkpoint loading requires active FSDP2 with at least two data-parallel ranks.")
                self._stream_load_transformer_dir = resolve_transformer_dir(self.pretrained_model_path, component_name=self.transformer_component)
                self.transformer = init_empty_minimax_h3_transformer(self.pretrained_model_path, **loader_kwargs)
            else:
                self.transformer = load_minimax_h3_transformer(self.pretrained_model_path, **loader_kwargs)
                self.transformer.to(self.device)

    def after_fsdp2_shard(self, config):
        """Load checkpoint tensors only after FSDP owns rank-local shards."""
        del config
        if not self._stream_load_pending:
            return
        stream_load_minimax_h3_transformer(
            self.transformer,
            self._stream_load_transformer_dir,
            device=self.device,
            lora_seed=self._stream_load_lora_seed,
            init_lora_weights=self._lora_init_weights(),
        )
        self._stream_load_pending = False

    def _load_vaes(self, config):
        try:
            from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        except ImportError as error:
            raise ImportError("MiniMax-H3 cache construction requires Diffusers with MiniMax-H3 video and audio VAEs.") from error

        local_files_only = bool(config.get("local_files_only", True))
        video_vae_subfolder = self._component_subfolder(
            self.pretrained_model_path,
            config.get("video_vae_subfolder"),
            candidates=("vae", "video_vae"),
        )
        self.video_vae = AutoencoderKLMiniMaxH3.from_pretrained(
            self.pretrained_model_path,
            subfolder=video_vae_subfolder,
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            self.pretrained_model_path,
            subfolder=config.get("audio_vae_subfolder", "audio_vae"),
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        for vae in (self.video_vae, self.audio_vae):
            vae.requires_grad_(False).eval()

        vae_attention_backend = config.get("vae_attention_backend")
        if vae_attention_backend not in {None, "torch_sdpa", "sdpa", "native"}:
            self.video_vae.set_attention_backend(vae_attention_backend)
        if self.cache_encoder_cpu_offload and self.device.type != "cpu":
            try:
                from accelerate import cpu_offload
            except ImportError as error:
                raise ImportError("MiniMax-H3 cache encoder CPU offload requires Accelerate.") from error
            self.video_vae.enable_group_offload(
                onload_device=self.device,
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
            )
            cpu_offload(self.audio_vae, execution_device=self.device, offload_buffers=True)
            self._managed_cache_encoders.update({"video", "audio"})
        else:
            self.video_vae.to(self.device)
            self.audio_vae.to(self.device)

        video_channels = int(self.video_vae.config.latent_channels)
        audio_channels = int(self.audio_vae.config.latent_channels)
        sampling_rate = int(self.audio_vae.config.sampling_rate)
        if video_channels != self.video_latent_channels:
            raise ValueError(f"MiniMax-H3 video_latent_channels does not match the video VAE: config={self.video_latent_channels}, checkpoint={video_channels}.")
        if audio_channels != self.audio_latent_channels:
            raise ValueError(f"MiniMax-H3 audio_latent_channels does not match the audio VAE: config={self.audio_latent_channels}, checkpoint={audio_channels}.")
        if sampling_rate != self.audio_sampling_rate:
            raise ValueError(f"MiniMax-H3 audio_sampling_rate does not match the audio VAE: config={self.audio_sampling_rate}, checkpoint={sampling_rate}.")

    @staticmethod
    def _component_subfolder(model_path, configured, candidates):
        if configured:
            return str(configured)
        root = Path(str(model_path)).expanduser()
        if root.is_dir():
            for candidate in candidates:
                if (root / candidate / "config.json").is_file():
                    return candidate
        return candidates[0]

    @contextmanager
    def _active_cache_encoder(self, encoder):
        name = "video" if encoder is self.video_vae else "audio" if encoder is self.audio_vae else None
        managed_offload = name in self._managed_cache_encoders
        should_offload = self.cache_encoder_cpu_offload and self.device.type != "cpu" and not managed_offload
        if should_offload:
            encoder.to(self.device)
        try:
            yield encoder
        finally:
            if should_offload:
                encoder.to("cpu")

    @staticmethod
    def _latent_statistics(vae, latents, shape):
        mean = latents.new_tensor(vae.config.latents_mean).view(shape)
        std = latents.new_tensor(vae.config.latents_std).view(shape)
        if bool((std == 0).any()):
            raise ValueError("MiniMax-H3 VAE latent standard deviations must be non-zero.")
        return mean, std

    @torch.inference_mode()
    def _encode_video_latents(self, video):
        with self._active_cache_encoder(self.video_vae) as video_vae:
            pixels = video.to(device=self.device, dtype=torch.float32)
            pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
            pixel_std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
            posterior = video_vae.encode((pixels - pixel_mean) / pixel_std).latent_dist
            latents = posterior.mode().float()
        mean, std = self._latent_statistics(self.video_vae, latents, shape=(1, -1, 1, 1, 1))
        return (latents - mean) / std

    @torch.inference_mode()
    def _encode_audio_latents(self, audio):
        with self._active_cache_encoder(self.audio_vae) as audio_vae:
            stereo_waveform = audio[0].unsqueeze(1).to(device=self.device, dtype=torch.float32)
            posterior = audio_vae.encode(stereo_waveform).latent_dist
            latents = posterior.mode().float().transpose(1, 2)
        mean, std = self._latent_statistics(self.audio_vae, latents, shape=(1, 1, -1))
        latents = (latents - mean) / std
        return latents.reshape(1, -1, self.audio_latent_channels)

    def reuse_frozen_components_from(self, source):
        super().reuse_frozen_components_from(source)
        self.video_vae = source.video_vae
        self.audio_vae = source.audio_vae
        self.condition_encoder = source.condition_encoder
        self._managed_cache_encoders = set(source._managed_cache_encoders)
        self._stream_load_lora_seed += 1

    def denoiser_module(self):
        return self.transformer

    @property
    def residual_head_feature_dims(self):
        return {
            "video": int(self.transformer.proj_out.in_features),
            "audio": int(self.transformer.audio_proj_out.in_features),
        }

    @property
    def residual_head_output_dims(self):
        return {
            "video": int(self.transformer.proj_out.out_features),
            "audio": int(self.transformer.audio_proj_out.out_features),
        }

    def residual_head_sigmas(self, sigma):
        capability = self.ensure_capabilities().require(DistributionMatchingCapability)
        return capability.residual_head_sigmas(sigma)

    def predict_velocity_with_features(self, latents, sigma, condition):
        capability = self.ensure_capabilities().require(DistributionMatchingCapability)
        return capability.predict_velocity_with_features(latents, sigma, condition)

    def transformer_forward_context(self):
        if self.use_autocast and self.device.type == "cuda" and self.running_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast("cuda", dtype=self.running_dtype)
        return nullcontext()

    def _lora_init_weights(self):
        # PEFT's True means Kaiming-uniform A (a=sqrt(5)) and zero B,
        # matching official PDMD. Keep every legacy run on Gaussian A.
        official = self.config.get("training", {}).get("dmd", {}).get("official_pdmd", False)
        return True if official else "gaussian"

    def add_lora(self, rank, alpha, target_modules):
        if not target_modules:
            target_modules = MiniMaxH3DistributionMatchingCapability._DEFAULT_LORA_TARGETS
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights=self._lora_init_weights(),
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
        sync_sequence_parallel_parameters(param for name, param in self.transformer.named_parameters() if "lora" in name and not param.is_meta)

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
        if not bool(torch.isin(text_token_tags, torch.tensor([0, 1], device=text_token_tags.device)).all()):
            raise ValueError("MiniMax-H3 base-task conditioner tags must be video=0 or text=1.")

        task = condition.get("task", "t2av")
        if isinstance(task, (list, tuple)):
            if len(task) != 1:
                raise ValueError(f"MiniMax-H3 cached task must contain one value, got {task!r}.")
            task = task[0]
        task = str(task)
        expected_anchors = {
            "t2av": (),
            "i2av": ("first",),
            "l2av": ("last",),
            "fl2av": ("first", "last"),
        }
        if task not in expected_anchors:
            raise ValueError(f"Unsupported MiniMax-H3 cached task {task!r}.")

        anchor_values = condition.get("keyframe_anchors")
        if anchor_values is None:
            anchors = ()
        elif torch.is_tensor(anchor_values):
            if anchor_values.ndim == 2:
                if anchor_values.shape[0] != 1:
                    raise ValueError("MiniMax-H3 currently requires data.train.batch_size=1.")
                anchor_values = anchor_values[0]
            if anchor_values.ndim != 1:
                raise ValueError(f"keyframe_anchors must be one-dimensional, got {tuple(anchor_values.shape)}.")
            names = {0: "first", 1: "last"}
            try:
                anchors = tuple(names[int(value)] for value in anchor_values.tolist())
            except KeyError as error:
                raise ValueError(f"Unknown MiniMax-H3 keyframe anchor code {error.args[0]}.") from error
        else:

            def normalize_anchor(value):
                # default_collate turns ("first", "last") into
                # [("first",), ("last",)] for batch_size=1.
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    value = value[0]
                return str(value)

            anchors = tuple(normalize_anchor(value) for value in anchor_values)
        if anchors != expected_anchors[task]:
            raise ValueError(f"MiniMax-H3 task={task} requires anchors={expected_anchors[task]}, got {anchors}.")

        def scalar_int(name):
            value = condition.get(name)
            if value is None:
                return None
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(f"MiniMax-H3 cached {name} must contain one value, got {tuple(value.shape)}.")
                return int(value.item())
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise ValueError(f"MiniMax-H3 cached {name} must contain one value, got {value!r}.")
                value = value[0]
            return int(value)

        target_height = scalar_int("target_height")
        target_width = scalar_int("target_width")
        target_num_frames = scalar_int("target_num_frames")
        if (target_height is None) != (target_width is None):
            raise ValueError("MiniMax-H3 cached target_height and target_width must be provided together.")

        condition_video_latents = condition.get("condition_video_latents")
        if anchors:
            if not torch.is_tensor(condition_video_latents):
                raise KeyError(f"MiniMax-H3 task={task} requires cached condition_video_latents.")
            if condition_video_latents.ndim == 3:
                if condition_video_latents.shape[0] != 1:
                    raise ValueError("MiniMax-H3 currently requires data.train.batch_size=1.")
                condition_video_latents = condition_video_latents[0]
            expected_dim = self.video_latent_channels * prod(self.patch_size)
            if condition_video_latents.ndim != 2 or condition_video_latents.shape[1] != expected_dim:
                raise ValueError(f"MiniMax-H3 condition_video_latents must have shape [rows, {expected_dim}], got {tuple(condition_video_latents.shape)}.")
            if target_height is not None and target_width is not None:
                rows_per_keyframe = (target_height // self.vae_spatial_scale_factor // self.patch_size[1]) * (target_width // self.vae_spatial_scale_factor // self.patch_size[2])
                expected_rows = len(anchors) * rows_per_keyframe
                if condition_video_latents.shape[0] != expected_rows:
                    raise ValueError(f"MiniMax-H3 task={task} expects {expected_rows} condition rows for {target_height}x{target_width}, got {condition_video_latents.shape[0]}.")
            condition_video_latents = condition_video_latents.to(self.device, dtype=torch.float32)
        elif condition_video_latents is not None and torch.is_tensor(condition_video_latents) and condition_video_latents.numel():
            raise ValueError("MiniMax-H3 t2av cache must not contain non-empty condition_video_latents.")
        else:
            condition_video_latents = None
        return {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=self.running_dtype),
            "text_token_tags": text_token_tags.to(self.device, dtype=torch.long),
            "task": task,
            "keyframe_anchors": anchors,
            "condition_video_latents": condition_video_latents,
            "target_height": target_height,
            "target_width": target_width,
            "target_num_frames": target_num_frames,
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

        video_latents = self._encode_video_latents(video)
        audio_latents = self._encode_audio_latents(audio)
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
