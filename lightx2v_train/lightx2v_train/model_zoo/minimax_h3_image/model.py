"""MiniMax-H3-Image with released-head PDMD and retained-head PDD adapters."""

import json
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model

from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.model_zoo.minimax_h3.condition_encoder import MiniMaxH3ConditionEncoder
from lightx2v_train.model_zoo.native.minimax_h3.sharded_loading import stream_load_minimax_h3_transformer
from lightx2v_train.model_zoo.native.minimax_h3_image import (
    HEAD_POLICY,
    PDDHeadSchedule,
    load_transformer,
    original_image_head_state,
    packed_t2i_layout,
    patchify_video_latents,
    resolve_transformer_dir,
    unpatchify_video_latents,
)
from lightx2v_train.model_zoo.pdd import PDDModelMixin
from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size, is_main_process
from lightx2v_train.runtime.fsdp import fsdp2_enabled
from lightx2v_train.runtime.sequence_parallel import sync_sequence_parallel_parameters
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .capability import MiniMaxH3ImageDistributionMatchingCapability


@MODEL_REGISTER("minimax_h3_image")
class MiniMaxH3ImageModel(PDDModelMixin, BaseModel):
    pipeline_cls = None

    def register_capabilities(self):
        super().register_capabilities()
        self.capabilities.register(DistributionMatchingCapability, MiniMaxH3ImageDistributionMatchingCapability(self))

    def load_components(self, *, load_transformer, load_vae, load_condition_encoder):
        config = self.config["model"]
        if get_sequence_parallel_world_size() != 1:
            raise ValueError("H3-Image PDMD currently uses FSDP data parallelism with sequence_parallel.size=1.")
        if config.get("sigma_head_policy", HEAD_POLICY) != HEAD_POLICY:
            raise ValueError(f"H3-Image sigma_head_policy must be {HEAD_POLICY!r}.")
        self.pretrained_model_path = config["pretrained_model_name_or_path"]
        self.image_head_mode = config.get("image_head_mode", "released")
        if self.image_head_mode not in {"released", "base"}:
            raise ValueError("model.image_head_mode must be 'released' or 'base'.")
        transformer_dir = resolve_transformer_dir(self.pretrained_model_path)
        with (transformer_dir / "config.json").open() as handle:
            transformer_config = json.load(handle)
        self.pdd_schedule = PDDHeadSchedule.from_model(self.pretrained_model_path)
        if int(transformer_config["num_pdd_heads"]) != len(self.pdd_schedule.sigmas) - 1:
            raise ValueError("H3-Image output-head count does not match pdd_config.json.")
        self.latent_channels = int(transformer_config["in_channels"])
        self.patch_size = tuple(transformer_config["patch_size"])
        self.text_dim = int(transformer_config["text_dim"])
        self.vae_scale_factor = 16
        self.latent_dtype = get_running_dtype(config.get("latent_dtype", "fp32"))
        self.transformer_param_dtype = get_running_dtype(config.get("transformer_param_dtype", "bf16"))
        self.transformer = None
        self.condition_encoder = None
        self._layout_key = None
        self._layout = None
        self._stream_load_pending = False
        self._stream_load_lora_seed = int(config.get("lora_init_seed", 0))
        if load_transformer:
            self._stream_load_pending = bool(self.config.get("distributed", {}).get("fsdp2", {}).get("stream_load_pretrained", False))
            if self._stream_load_pending and not fsdp2_enabled(self.config):
                raise ValueError("H3-Image streamed loading requires active FSDP2 with at least two data-parallel ranks.")
            self._stream_load_transformer_dir = transformer_dir
            self.transformer = self._load_transformer(meta=self._stream_load_pending)
            if not self._stream_load_pending:
                self.transformer.to(self.device)
        if load_vae:
            from diffusers import AutoencoderKLMiniMaxH3Image

            self.vae = (
                AutoencoderKLMiniMaxH3Image.from_pretrained(self.pretrained_model_path, subfolder="vae_diffusers", torch_dtype=torch.float32, local_files_only=True, low_cpu_mem_usage=True)
                .to(self.device)
                .requires_grad_(False)
                .eval()
            )
            self.vae_config = self.vae.config
        if load_condition_encoder:
            self.condition_encoder = MiniMaxH3ConditionEncoder(
                self.pretrained_model_path,
                device=self.device,
                dtype=self.running_dtype,
                cpu_offload=bool(config.get("cache_condition_encoder_cpu_offload", False)),
                attention_backend=config.get("condition_attention_backend", "torch_sdpa"),
                text_encoder_layer=50,
            )

    def _load_transformer(self, *, meta):
        return load_transformer(
            self.pretrained_model_path, torch_dtype=self.transformer_param_dtype, meta=meta, attention_backend=self.config["model"].get("attention_backend"), image_head_mode=self.image_head_mode
        )

    def after_fsdp2_shard(self, config):
        del config
        if self._stream_load_pending:
            extra_state = original_image_head_state(self.pretrained_model_path) if self.image_head_mode == "base" else None
            stream_load_minimax_h3_transformer(
                self.transformer,
                self._stream_load_transformer_dir,
                device=self.device,
                lora_seed=self._stream_load_lora_seed,
                init_lora_weights=True,
                tensor_transform=self.transform_pdd_pretrained_tensor,
                extra_state=extra_state,
            )
            self._stream_load_pending = False

    def reuse_frozen_components_from(self, source):
        super().reuse_frozen_components_from(source)
        self.condition_encoder = source.condition_encoder
        self._stream_load_lora_seed += 1

    def denoiser_module(self):
        return self.transformer

    def add_lora(self, rank, alpha, target_modules, *, rank_pattern=None, alpha_pattern=None):
        if not target_modules:
            target_modules = self.ensure_capabilities().require(DistributionMatchingCapability).default_lora_target_modules
        config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules, rank_pattern=rank_pattern or {}, alpha_pattern=alpha_pattern or {}, init_lora_weights=True)
        self.transformer = inject_adapter_in_model(config, self.transformer, adapter_name="default")
        sync_sequence_parallel_parameters(parameter for name, parameter in self.transformer.named_parameters() if ".lora_" in name and not parameter.is_meta)

    def prepare_text_condition(self, condition):
        embeddings = condition["prompt_embeds"]
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        tags = condition["text_token_tags"].reshape(-1)
        if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[-1] != self.text_dim or tags.shape != (embeddings.shape[1],) or not bool((tags == 1).all()):
            raise ValueError(f"H3-Image T2I expects prompt_embeds [1,L,{self.text_dim}] and L text tags equal to 1.")
        return {"prompt_embeds": embeddings.to(device=self.device, dtype=self.running_dtype), "text_token_tags": tags.to(device=self.device, dtype=torch.long)}

    def encode_prompt_condition(self, prompt):
        if self.condition_encoder is None:
            raise RuntimeError("H3-Image condition encoder is unloaded; provide the preprocessed text cache.")
        return self.prepare_text_condition(self.condition_encoder.encode(prompt))

    def encode_condition(self, sample):
        return self.encode_prompt_condition(sample["conditioning"]["prompt"])

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        if noisy_latent.ndim != 5 or tuple(noisy_latent.shape[:3]) != (1, self.latent_channels, 1):
            raise ValueError(f"H3-Image latent must be [1,{self.latent_channels},1,H,W], got {tuple(noisy_latent.shape)}.")
        length = condition["prompt_embeds"].shape[-2]
        key = (length, *noisy_latent.shape[-2:], noisy_latent.device)
        if key != self._layout_key:
            self._layout = packed_t2i_layout(length, *noisy_latent.shape[-2:], patch_size=self.patch_size, device=noisy_latent.device)
            self._layout_key = key
        return {"rows": patchify_video_latents(noisy_latent, self.patch_size).unsqueeze(0), "shape": tuple(noisy_latent.shape), "layout": self._layout}

    def denoise(self, denoiser_input, timesteps, condition):
        # The native flow is clean-minus-noise; Tensor PDMD uses the opposite.
        return -self._forward_image(denoiser_input, timesteps, condition).float()

    def _forward_image(self, denoiser_input, timesteps, condition, *, feature_block=None):
        sigma = torch.as_tensor(timesteps, device=self.device, dtype=torch.float32).reshape(-1)
        head = self.pdd_schedule.head_index(sigma) if self.image_head_mode == "released" else 0
        # Use the physical sigma directly, with no timestep *1000 or shift.
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self.transformer(
                hidden_states=denoiser_input["rows"],
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=1 - sigma,
                pdd_step_index=head,
                **denoiser_input["layout"],
                return_dict=False,
                return_features_block=feature_block,
            )[0]

    def predict_dmad_features(self, latents, sigma, condition, feature_block=49):
        """Differentiable target image tokens before output normalization.

        DMD2's discriminator shares the fake backbone. Its generator loss needs
        input gradients even while that backbone's parameters are frozen.
        """
        denoiser_input = self.prepare_denoiser_input(latents, condition=condition)
        features = self._forward_image(denoiser_input, sigma, condition, feature_block=feature_block)
        return {"image": features}

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return unpatchify_video_latents(prediction, denoiser_input["shape"], self.patch_size)

    def pdd_output_layer_paths(self):
        if self.image_head_mode != "base":
            raise ValueError("Image PDD requires explicit model.image_head_mode=base; released six fused heads remain the PDMD path.")
        return {"image": "proj_out"}

    def pdd_sigma_schedule(self, primary_sigma):
        return torch.as_tensor(primary_sigma, device=self.device)

    def predict_pdd_teacher(self, latents, sigma, condition):
        return self.predict_denoiser_output(latents, sigma, condition)

    def predict_pdd_heads(self, latents, start_index, condition, sigma):
        if not 0 <= int(start_index) < self.pdd_steps:
            raise ValueError("PDD start_index must address an interval in the configured base grid.")
        denoiser_input = self.prepare_denoiser_input(latents, condition=condition)
        rows = self.denoise(denoiser_input, sigma, condition)
        batch, _, channels = rows.shape
        rows = rows.unflatten(-1, (self.pdd_steps, channels // self.pdd_steps)).movedim(-2, 1).flatten(0, 1)
        shape = (batch * self.pdd_steps, *latents.shape[1:])
        return unpatchify_video_latents(rows, shape, self.patch_size).unflatten(0, (batch, self.pdd_steps))

    def fsdp2_shard_plan(self, fsdp_config):
        reshard = fsdp_config.get("reshard_after_forward", {})
        projections = [self.transformer.proj_in, self.transformer.time_embedder]
        if self.image_head_mode == "base":
            projections.append(self.transformer.proj_out)
        return [
            {
                "modules": projections,
                "reshard_after_forward": reshard.get("block_reshard", True),
                "mixed_precision": {"param_dtype": "fp32", "reduce_dtype": "fp32", "output_dtype": None, "cast_forward_inputs": False},
            },
            {
                # All heads share one FSDP group: ranks independently sample
                # score sigmas and can select different heads in the same step.
                "module": self.transformer.pdd_proj_out,
                "reshard_after_forward": reshard.get("block_reshard", True),
                "mixed_precision": {"param_dtype": "fp32", "reduce_dtype": "fp32", "output_dtype": None, "cast_forward_inputs": False},
            },
            {"modules": [*self.transformer.token_refiner.refiner_blocks, *self.transformer.transformer_blocks], "reshard_after_forward": reshard.get("block_reshard", True)},
            {"module": self.transformer, "reshard_after_forward": reshard.get("root_reshard", False)},
        ]

    def prepare_infer_latents(self, height, width, generator=None):
        if height % 32 or width % 32:
            raise ValueError("H3-Image image dimensions must be divisible by 32.")
        return torch.randn((1, self.latent_channels, 1, height // 16, width // 16), device=self.device, dtype=torch.float32, generator=generator)

    @torch.no_grad()
    def decode_latent(self, latent):
        values = latent.squeeze(2).float()
        mean = values.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1)
        std = values.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1)
        image = self.vae.decode(values * std + mean, return_dict=False)[0].float()
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (image * std + mean).clamp(0, 1)

    def save_lora_weights(self, save_dir, adapter_name=None, weights_subdir=None, **kwargs):
        super().save_lora_weights(save_dir, adapter_name=adapter_name, weights_subdir=weights_subdir, **kwargs)
        if is_main_process():
            directory = Path(save_dir) / weights_subdir if weights_subdir else Path(save_dir)
            training = self.config.get("training", {})
            if training.get("method") == "pdd":
                metadata = {
                    "model": "minimax_h3_image",
                    "method": "pdd",
                    "image_head_mode": self.image_head_mode,
                    "requires_pdd_heads": True,
                    "sampling_config": "pdd_sampling.json",
                    "head_weights": "pdd_heads.safetensors",
                    "pdd": training["pdd"],
                    "velocity": "noise_minus_clean",
                }
            else:
                dmd = training.get("dmd", {})
                metadata = {"model": "minimax_h3_image", **self.pdd_schedule.metadata(), "student_sigmas": dmd.get("official_rollout_sigmas"), "native_velocity": "clean_minus_noise"}
            (directory / "h3_image_sampling.json").write_text(json.dumps(metadata, indent=2) + "\n")
