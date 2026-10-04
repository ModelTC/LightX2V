"""Distribution-matching capability for MiniMax-H3 T2AV."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from lightx2v_train.model_capabilities import DistributionMatchingProfile, LossResult
from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistributionMatchingCapability,
    _require_single_prompt,
    _require_singleton_tensor,
)
from lightx2v_train.model_zoo.native.minimax_h3 import (
    KEYFRAME_NOISE_AUG,
    MiniMaxH3SparseOnlyAttnProcessor,
    MiniMaxH3StudentSLAConfig,
    audio_latent_num_frames,
    build_packed_sequence,
    build_ref2av_packed_sequence,
    build_row_timesteps,
    install_minimax_h3_student_sla,
    keyframe_condition_noise,
    patchify_video_latents,
    video_latent_num_frames,
)
from lightx2v_train.runtime.distributed import (
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_world_size,
    reduce_mean,
)
from lightx2v_train.trainers.dmd.math import dmd_loss_with_stats
from lightx2v_train.utils.generation_shapes import resolve_generation_shape

from ..adaptive_video_regularization import AdaptiveVideoRegularizer
from ..memory_guard import MiniMaxH3MemoryGuard
from .common import MiniMaxH3JointLatents, MiniMaxH3LatentShape


@dataclass(frozen=True)
class MiniMaxH3DistributionMatchingOptions:
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    video_flow_shift: float = 6.0
    audio_flow_shift: float = 3.0
    audio_dmd_loss_weight: float = 1.0
    geometry_from_metadata: bool = False
    fixed_num_frames: int | None = None
    allowed_resolutions: tuple[tuple[int, int], ...] = ()
    layout_cache_size: int = 16
    projected_dmd: bool = False
    dmd_normalization: bool = True
    dmd_normalization_epsilon: float = 0.0
    dmd_reduction: str = "mean"
    student_sparse_attention: Mapping | None = None
    adaptive_video_regularization: Mapping | None = None
    memory_guard: Mapping | None = None

    @classmethod
    def from_mapping(cls, config: Mapping | None) -> "MiniMaxH3DistributionMatchingOptions":
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("model.capabilities.distribution_matching must be a mapping.")
        options = cls(
            video_loss_weight=float(config.get("video_loss_weight", 1.0)),
            audio_loss_weight=float(config.get("audio_loss_weight", 1.0)),
            video_flow_shift=float(config.get("video_flow_shift", 6.0)),
            audio_flow_shift=float(config.get("audio_flow_shift", 3.0)),
            audio_dmd_loss_weight=float(config.get("audio_dmd_loss_weight", config.get("audio_loss_weight", 1.0))),
            geometry_from_metadata=bool(config.get("geometry_from_metadata", False)),
            fixed_num_frames=None if config.get("fixed_num_frames") is None else int(config["fixed_num_frames"]),
            allowed_resolutions=tuple(tuple(map(int, value)) for value in config.get("allowed_resolutions", ())),
            layout_cache_size=int(config.get("layout_cache_size", 16)),
            projected_dmd=bool(config.get("projected_dmd", False)),
            dmd_normalization=bool(config.get("dmd_normalization", True)),
            dmd_normalization_epsilon=float(config.get("dmd_normalization_epsilon", 0.0)),
            dmd_reduction=str(config.get("dmd_reduction", "mean")),
            student_sparse_attention=config.get("student_sparse_attention"),
            adaptive_video_regularization=config.get("adaptive_video_regularization"),
            memory_guard=config.get("memory_guard"),
        )
        if options.video_flow_shift <= 0 or options.audio_flow_shift <= 0:
            raise ValueError("MiniMax-H3 video_flow_shift and audio_flow_shift must be positive.")
        if options.video_loss_weight < 0 or options.audio_loss_weight < 0:
            raise ValueError("MiniMax-H3 video_loss_weight and audio_loss_weight cannot be negative.")
        if options.video_loss_weight == 0 and options.audio_loss_weight == 0:
            raise ValueError("At least one MiniMax-H3 modality loss weight must be non-zero.")
        if options.audio_dmd_loss_weight < 0 or options.dmd_normalization_epsilon < 0:
            raise ValueError("H3 DMD audio weight and normalization epsilon must be non-negative.")
        if options.layout_cache_size < 1 or options.dmd_reduction not in {"mean", "sum"}:
            raise ValueError("H3 layout_cache_size must be positive and dmd_reduction must be mean or sum.")
        if options.fixed_num_frames is not None:
            video_latent_num_frames(options.fixed_num_frames)
        if any(len(value) != 2 or min(value) <= 0 or any(side % 32 for side in value) for value in options.allowed_resolutions):
            raise ValueError("H3 allowed_resolutions must contain positive [height, width] multiples of 32.")
        return options


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply H3's rational flow shift to an unshifted noise level."""

    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _expand_sigma(sigma: torch.Tensor, ndim: int) -> torch.Tensor:
    if sigma.ndim == 0:
        sigma = sigma.reshape(1)
    return sigma.reshape(sigma.shape[0], *([1] * (ndim - 1)))


class MiniMaxH3DistributionMatchingCapability(GenericDistributionMatchingCapability):
    """H3-specific operations consumed by the framework's generic DMD loop.

    H3 jointly denoises packed video and stereo-audio tokens. It also uses a
    clean-ward velocity (``x0 - noise``), while most LightX2V models use the
    opposite flow direction. Keeping those details here lets ``DmdTrainer``
    manage roles, rollout, optimization, checkpointing, and FSDP unchanged.
    """

    _DEFAULT_LORA_TARGETS = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    )
    _PROFILE = DistributionMatchingProfile(
        supported_training_methods=frozenset({"dmd"}),
        supports_guidance=False,
        supports_ida=False,
        supports_diversity=False,
        supports_real_data_fake=False,
        supports_warped_denoising_schedule=False,
        default_latent_dtype=torch.float32,
    )

    def __init__(self, model, options: Mapping | None = None) -> None:
        super().__init__(model)
        options = MiniMaxH3DistributionMatchingOptions.from_mapping(options)
        self.video_weight = options.video_loss_weight
        self.audio_weight = options.audio_loss_weight
        self.audio_dmd_weight = options.audio_dmd_loss_weight
        self.options = options
        self.video_shift = options.video_flow_shift
        self.audio_shift = options.audio_flow_shift
        self._layout_cache = OrderedDict()
        self._last_dmd_metrics = {}
        self._training_roles = None
        steps = int(model.config.get("training", {}).get("dmd", {}).get("num_inference_steps", 1))
        self.adv_regularizer = AdaptiveVideoRegularizer(options.adaptive_video_regularization, steps)
        self.student_sla = MiniMaxH3StudentSLAConfig.from_mapping(options.student_sparse_attention)
        self.memory_guard = MiniMaxH3MemoryGuard(options.memory_guard, model)
        if model.transformer_component == "transformer_ref" and self.adv_regularizer.enabled:
            raise ValueError("H3 Ref2AV uses condition-only DMD; adaptive video regularization must be disabled.")

    @property
    def profile(self) -> DistributionMatchingProfile:
        return self._PROFILE

    @property
    def default_negative_prompt(self):
        return ""

    @property
    def default_lora_target_modules(self):
        return self._DEFAULT_LORA_TARGETS

    @property
    def generation_shape_dimensions(self) -> int:
        return 3

    def latent_shape(
        self,
        batch,
        generation_shapes,
        broadcast,
    ):
        prompt = batch.get("conditioning", {}).get("prompt", "")
        _require_single_prompt(prompt)
        if self.options.geometry_from_metadata:
            metadata = batch.get("meta", {})
            condition = batch.get("conditioning", {}).get("positive", {})

            def dimension(name, fallback=None):
                value = metadata.get(name, condition.get(name, fallback))
                if value is None:
                    raise KeyError(f"H3 metadata requires {name}.")
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        raise ValueError(f"H3 metadata {name} must contain one value.")
                    value = value.item()
                if isinstance(value, (list, tuple)):
                    if len(value) != 1:
                        raise ValueError(f"H3 metadata {name} must contain one value.")
                    value = value[0]
                return int(broadcast(int(value)))

            num_frames = self.options.fixed_num_frames or dimension("target_num_frames", metadata.get("num_frames"))
            height, width = dimension("target_height"), dimension("target_width")
        else:
            num_frames, height, width = resolve_generation_shape(
                generation_shapes,
                batch,
                expected_dimensions=self.generation_shape_dimensions,
                broadcast=broadcast,
            )
        if self.options.allowed_resolutions and (height, width) not in self.options.allowed_resolutions:
            raise ValueError(f"H3 sample resolution {height}x{width} is outside allowed_resolutions.")
        size_multiple = self.model.vae_spatial_scale_factor * self.model.patch_size[1]
        width_multiple = self.model.vae_spatial_scale_factor * self.model.patch_size[2]
        if height % size_multiple or width % width_multiple:
            raise ValueError(f"MiniMax-H3 height/width must be divisible by the VAE and patch scales ({size_multiple}, {width_multiple}), got {height}x{width}.")

        latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // self.model.vae_spatial_scale_factor
        latent_width = width // self.model.vae_spatial_scale_factor
        patch_t, patch_h, patch_w = self.model.patch_size
        if patch_t != 1:
            raise ValueError(f"MiniMax-H3 T2AV requires temporal patch size 1, got {self.model.patch_size}.")
        video_rows = latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
        video_dimension = self.model.video_latent_channels * patch_t * patch_h * patch_w
        audio_latents = audio_latent_num_frames(num_frames)
        return MiniMaxH3LatentShape(
            num_frames=num_frames,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            audio_latents=audio_latents,
            video_tokens=(1, video_rows, video_dimension),
            audio_tokens=(1, audio_latents * 2, self.model.audio_latent_channels),
        )

    def encode_conditions(
        self,
        batch,
        negative_prompt,
        guidance_scale,
        broadcast,
    ):
        del negative_prompt
        if guidance_scale != 1.0:
            raise ValueError("MiniMax-H3 is guidance-distilled; training.teacher.guidance_scale must be 1.0.")
        conditioning = batch.get("conditioning", {})
        cached_condition = conditioning.get("positive")
        with torch.no_grad():
            if cached_condition is None:
                condition = self.model.encode_condition(batch)
            else:
                # Reference VAE rows stay FP32; only the text encoder rows
                # are converted to the transformer's running dtype.
                cached_condition = {**conditioning.get("shared", {}), **cached_condition}
                condition = self.model.prepare_text_condition(cached_condition)
        condition = self._prepare_condition_for_rollout(condition, broadcast)
        return broadcast(condition), None

    def predict_velocity(self, latents, sigma, condition):
        self._validate_latents(latents)
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        layout = self._layout(condition, latents.shape)
        timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_sigma,
            audio_sigma,
        )
        transformer_video = latents.video
        transformer_audio = latents.audio
        if layout.num_condition_video_rows:
            rows = condition["noised_condition_video_latents"]
            if rows.shape[0] != layout.num_condition_video_rows:
                raise ValueError("H3 cached visual rows do not match the packed sequence.")
            transformer_video = torch.cat((rows.to(latents.video).unsqueeze(0), latents.video), dim=1)
        if layout.num_condition_audio_rows:
            rows = condition["condition_audio_latents"]
            if rows.shape[0] != layout.num_condition_audio_rows:
                raise ValueError("H3 cached audio rows do not match the packed sequence.")
            transformer_audio = torch.cat((rows.to(latents.audio).unsqueeze(0), latents.audio), dim=1)
        with self.model.transformer_forward_context():
            prediction = self.model.denoiser_module()(
                hidden_states=transformer_video,
                audio_hidden_states=transformer_audio,
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=timesteps.to(self.device),
                timestep_indices=timestep_indices.to(self.device),
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )
        if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
            raise TypeError("MiniMax-H3 transformer must return (video_velocity, audio_velocity) when return_dict=False.")
        return MiniMaxH3JointLatents(
            video=prediction[0][:, layout.num_condition_video_rows :],
            audio=prediction[1][:, layout.num_condition_audio_rows :],
            shape=latents.shape,
        )

    def predict_guided_velocity(
        self,
        latents,
        sigma,
        condition,
        negative_condition,
        guidance_scale,
        cfg_norm,
    ):
        del cfg_norm
        if negative_condition is not None or guidance_scale != 1.0:
            raise ValueError("MiniMax-H3 has no unconditional branch and only supports guidance_scale=1.0.")
        return self.predict_velocity(latents, sigma, condition)

    def initial_latents(self, latent_shape, dtype, broadcast):
        if latent_shape.video_tokens[0] != 1 or latent_shape.audio_tokens[0] != 1:
            raise ValueError(f"MiniMax-H3 DMD requires physical batch size 1, got {latent_shape}.")
        video = broadcast(torch.randn(latent_shape.video_tokens, device=self.device, dtype=dtype))
        audio = broadcast(torch.randn(latent_shape.audio_tokens, device=self.device, dtype=dtype))
        return MiniMaxH3JointLatents(video, audio, latent_shape)

    @staticmethod
    def latent_hw(latent_shape):
        del latent_shape
        return None

    @staticmethod
    def random_noise_like(latents, dtype, broadcast):
        return MiniMaxH3JointLatents(
            broadcast(torch.randn_like(latents.video, dtype=dtype)),
            broadcast(torch.randn_like(latents.audio, dtype=dtype)),
            latents.shape,
        )

    def add_noise(self, scheduler, latents, noise, sigma):
        del scheduler
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        return MiniMaxH3JointLatents(
            self._mix_noise(latents.video, noise.video, video_sigma),
            self._mix_noise(latents.audio, noise.audio, audio_sigma),
            latents.shape,
        )

    @staticmethod
    def training_target(latents, noise):
        # H3's time is t=1-sigma, so its velocity points from noise to x0.
        return MiniMaxH3JointLatents(
            latents.video.float() - noise.video.float(),
            latents.audio.float() - noise.audio.float(),
            latents.shape,
        )

    def step(self, scheduler, velocity, step_index, sample):
        sigma = scheduler.sigma_at(step_index, device=self.device, dtype=torch.float32)
        sigma_next = scheduler.sigma_at(int(step_index) + 1, device=self.device, dtype=torch.float32)
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        video_sigma_next, audio_sigma_next = self._modality_sigmas(sigma_next)
        return (
            MiniMaxH3JointLatents(
                self._cleanward_step(sample.video, velocity.video, video_sigma, video_sigma_next),
                self._cleanward_step(sample.audio, velocity.audio, audio_sigma, audio_sigma_next),
                sample.shape,
            ),
            MiniMaxH3JointLatents(
                self._cleanward_x0(sample.video, velocity.video, video_sigma),
                self._cleanward_x0(sample.audio, velocity.audio, audio_sigma),
                sample.shape,
            ),
        )

    def x0_from_velocity(self, sample, velocity, sigma):
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        return MiniMaxH3JointLatents(
            self._cleanward_x0(sample.video, velocity.video, video_sigma),
            self._cleanward_x0(sample.audio, velocity.audio, audio_sigma),
            sample.shape,
        )

    def regression_loss(self, prediction, target):
        video_loss = F.mse_loss(prediction.video.float(), target.video.float())
        audio_loss = F.mse_loss(prediction.audio.float(), target.audio.float())
        return self.video_weight * video_loss + self.audio_weight * audio_loss

    def dmd_loss(self, latents, fake_x0, teacher_x0):
        kwargs = dict(
            normalize=self.options.dmd_normalization,
            normalization_epsilon=self.options.dmd_normalization_epsilon,
            reduction=self.options.dmd_reduction,
            projected=self.options.projected_dmd,
        )
        video_loss, video_normalizer, video_rms = dmd_loss_with_stats(
            latents.video,
            fake_x0.video,
            teacher_x0.video,
            **kwargs,
        )
        audio_loss, audio_normalizer, audio_rms = dmd_loss_with_stats(
            latents.audio,
            fake_x0.audio,
            teacher_x0.audio,
            **kwargs,
        )
        self._last_dmd_metrics = {
            "dmd_video": (self.video_weight * video_loss).detach(),
            "dmd_audio": (self.audio_dmd_weight * audio_loss).detach(),
            "dmd_video_normalizer": video_normalizer,
            "dmd_audio_normalizer": audio_normalizer,
            "dmd_video_direction_rms": video_rms,
            "dmd_audio_direction_rms": audio_rms,
        }
        return self.video_weight * video_loss + self.audio_dmd_weight * audio_loss

    def dmd_metrics(self):
        return self._last_dmd_metrics

    def prepare_role(self, role):
        if role != "student" or not self.student_sla.enabled:
            return
        if get_sequence_parallel_world_size() > 1:
            raise ValueError("H3 student sparse attention requires sequence parallel size 1.")
        install_minimax_h3_student_sla(self.model.denoiser_module(), self.student_sla)

    def validate_training_roles(self, student_model, fake_model, teacher_model):
        roles = {"student": student_model, "fake": fake_model, "teacher": teacher_model}
        if self.student_sla.enabled:
            for role in ("fake", "teacher"):
                if any(isinstance(block.attn.processor, MiniMaxH3SparseOnlyAttnProcessor) for block in roles[role].denoiser_module().transformer_blocks):
                    raise RuntimeError(f"H3 {role} attention must remain dense when student sparse attention is enabled.")
        self._training_roles = roles
        self.memory_guard.validate(roles, self.adv_regularizer.enabled, self.student_sla.enabled)

    def on_iteration_start(self, current_iter):
        del current_iter
        self.memory_guard.start()

    def on_iteration_end(self, current_iter):
        del current_iter
        self.memory_guard.finish()

    def after_optimizer_step(self, role):
        if role == "student":
            self.adv_regularizer.commit_regression_ema()

    def extra_training_state(self):
        return {"adaptive_video_regularization": self.adv_regularizer.state_dict()} if self.adv_regularizer.enabled else {}

    def load_extra_training_state(self, state):
        saved = state.get("adaptive_video_regularization")
        if self.adv_regularizer.enabled:
            if saved is None:
                raise RuntimeError("H3 checkpoint has no adaptive video regression state.")
            self.adv_regularizer.load_state_dict(saved)
        elif saved is not None:
            raise RuntimeError("H3 checkpoint contains adaptive video regression state but the option is disabled.")

    def extra_checkpoint_metadata(self):
        metadata = {
            "minimax_h3_distribution_matching": {
                "transformer_component": self.model.transformer_component,
                "video_flow_shift": self.video_shift,
                "audio_flow_shift": self.audio_shift,
                "video_loss_weight": self.video_weight,
                "audio_loss_weight": self.audio_weight,
                "audio_dmd_loss_weight": self.audio_dmd_weight,
                "projected_dmd": self.options.projected_dmd,
                "adaptive_video_regularization": self.options.adaptive_video_regularization or {"enabled": False},
            },
            "minimax_h3_per_modality_normalization": {
                "enabled": self.options.dmd_normalization,
                "epsilon": self.options.dmd_normalization_epsilon,
                "reduction": self.options.dmd_reduction,
            },
            "minimax_h3_target_geometry": {
                "geometry_from_metadata": self.options.geometry_from_metadata,
                "fixed_num_frames": self.options.fixed_num_frames,
                "fallback_num_frames": self.model.config.get("training", {}).get("dmd", {}).get("num_frames", 124),
                "allowed_resolutions": [list(value) for value in sorted(self.options.allowed_resolutions)],
            },
            "minimax_h3_parallel_topology": {
                "world_size": int(get_world_size()),
                "sequence_parallel_size": int(get_sequence_parallel_world_size()),
                "fsdp2_size": int(get_data_parallel_world_size()),
                "all_denoisers_fsdp2": bool(self._training_roles) and all(model.is_fsdp2_wrapped() for model in self._training_roles.values()),
            },
        }
        if self.student_sla.enabled:
            metadata["student_sparse_attention"] = self.student_sla.checkpoint_metadata()
        return metadata

    @staticmethod
    def _single_metadata_value(meta, name):
        value = meta.get(name)
        if value is None:
            raise KeyError(f"ADV adaptive regression requires {name} in the full teacher-latent metadata.")
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"H3 ADV currently requires batch_size=1; got {name}={value!r}.")
            value = value[0]
        return value

    @staticmethod
    def _payload_flag(payload, name, default=False):
        value = payload.get(name, default)
        if torch.is_tensor(value):
            return bool(value.detach().reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            return bool(value[0]) if value else bool(default)
        return bool(value)

    def _load_real_av_latents(self, sample, latent_shape):
        inputs = sample.get("inputs", {})
        video_payload = inputs.get("video_latents")
        if video_payload is None:
            path = Path(str(self._single_metadata_value(sample.get("meta", {}), "video_latent_path")))
            video_payload = torch.load(path, map_location="cpu", weights_only=True)
        if torch.is_tensor(video_payload):
            video = video_payload
        elif isinstance(video_payload, dict) and torch.is_tensor(video_payload.get("latents")):
            if not self._payload_flag(video_payload, "normalized", False):
                raise ValueError("H3 ADV video latent payload must be marked normalized=true.")
            video = video_payload["latents"]
        else:
            raise TypeError("H3 ADV video_latents must be a tensor or a dict containing tensor 'latents'.")
        if video.ndim == 4:
            video = video.unsqueeze(0)

        audio = inputs.get("audio_latents")
        if audio is None:
            path = Path(str(self._single_metadata_value(sample.get("meta", {}), "audio_latent_path")))
            audio = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(audio, dict):
            audio = audio.get("latents")
        if not torch.is_tensor(audio):
            raise TypeError("H3 ADV audio_latents must be a tensor.")
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)

        expected_video = (
            1,
            self.model.video_latent_channels,
            latent_shape.latent_frames,
            latent_shape.latent_height,
            latent_shape.latent_width,
        )
        expected_audio = (
            1,
            2,
            self.model.audio_latent_channels,
            latent_shape.audio_latents,
        )
        if tuple(video.shape) != expected_video:
            raise ValueError(f"H3 ADV video latent shape {tuple(video.shape)} does not match sample geometry {expected_video}.")
        if tuple(audio.shape) != expected_audio:
            raise ValueError(f"H3 ADV audio latent shape {tuple(audio.shape)} does not match sample geometry {expected_audio}.")

        video = video.to(device=self.model.device, dtype=torch.float32)
        audio = audio.to(device=self.model.device, dtype=torch.float32)
        video_rows = patchify_video_latents(video, self.model.patch_size).unsqueeze(0)
        audio_rows = (
            audio.permute(0, 1, 3, 2)
            .reshape(
                1,
                latent_shape.audio_latents * 2,
                self.model.audio_latent_channels,
            )
            .contiguous()
        )
        if tuple(video_rows.shape) != latent_shape.video_tokens:
            raise ValueError(f"H3 ADV packed video shape {tuple(video_rows.shape)} != {latent_shape.video_tokens}.")
        if tuple(audio_rows.shape) != latent_shape.audio_tokens:
            raise ValueError(f"H3 ADV packed audio shape {tuple(audio_rows.shape)} != {latent_shape.audio_tokens}.")
        return MiniMaxH3JointLatents(video_rows, audio_rows, latent_shape)

    def legacy_extra_checkpoint_metadata(self):
        metadata = self.extra_checkpoint_metadata()
        metadata.pop("minimax_h3_parallel_topology")
        metadata.pop("minimax_h3_target_geometry")
        if "student_sparse_attention" in metadata:
            metadata["student_sparse_attention"] = {"enabled": False}
        metadata["minimax_h3_distribution_matching"]["projected_dmd"] = False
        metadata["minimax_h3_per_modality_normalization"] = {
            "enabled": True,
            "epsilon": 0.0,
            "reduction": "mean",
        }
        return metadata

    def student_regularization(self, generated, sample, condition, scheduler, broadcast):
        regularizer = self.adv_regularizer
        if not regularizer.enabled:
            return None
        loss = generated.video.float().new_zeros(())
        metrics = {}
        if regularizer.temporal_enabled:
            temporal, raw, variance = regularizer.temporal_loss(generated.video, generated.shape.latent_frames)
            loss = loss + temporal
            metrics.update(
                adv_temp_raw=raw.detach(), adv_temp_weighted=temporal.detach(), adv_motion_metric=variance.detach(), adv_temp_active=(raw.detach() >= regularizer.temporal.loss_threshold).float()
            )
        if regularizer.regression_enabled or regularizer.audio_regression_enabled:
            if sample is None:
                raise ValueError("H3 adaptive flow regression requires teacher video and audio latents.")
            clean = self._load_real_av_latents(sample, generated.shape)
            clean = MiniMaxH3JointLatents(broadcast(clean.video), broadcast(clean.audio), clean.shape)
            index = int(broadcast(torch.randint(regularizer.num_inference_steps, (1,), device=self.device)).item())
            sigma = torch.tensor(1.0 - index / regularizer.num_inference_steps, device=self.device)
            noise = self.random_noise_like(clean, torch.float32, broadcast)
            noised = self.add_noise(scheduler, clean, noise, sigma)
            velocity = self.predict_velocity(noised, sigma, condition)
            target = self.training_target(clean, noise)
            metrics["adv_reg_step"] = loss.new_tensor(float(index))
            if regularizer.regression_enabled:
                raw = F.mse_loss(velocity.video.float(), target.video)
                global_mean = float(reduce_mean(float(raw.detach().item())))
                baseline = regularizer.regression_ema[index]
                weighted, weight = regularizer.regression_loss(raw, index, global_mean)
                loss = loss + weighted
                metrics.update(adv_reg_raw=raw.detach(), adv_reg_weight=weight.detach(), adv_reg_weighted=weighted.detach(), adv_reg_ema=loss.new_tensor(global_mean if baseline is None else baseline))
            if regularizer.audio_regression_enabled:
                raw = F.mse_loss(velocity.audio.float(), target.audio)
                weighted = regularizer.audio_regression.weight * raw
                loss = loss + weighted
                metrics.update(adv_audio_reg_raw=raw.detach(), adv_audio_reg_weighted=weighted.detach())
        return LossResult(loss, metrics)

    @staticmethod
    def detach(value):
        return MiniMaxH3JointLatents(
            value.video.detach(),
            value.audio.detach(),
            value.shape,
        )

    @staticmethod
    def to_dtype(value, dtype):
        return MiniMaxH3JointLatents(
            value.video.to(dtype=dtype),
            value.audio.to(dtype=dtype),
            value.shape,
        )

    def extract_real_latents(self, batch, dtype, broadcast):
        del batch, dtype, broadcast
        raise ValueError("MiniMax-H3 joint DMD does not support real-data fake loss.")

    def _modality_sigmas(self, sigma):
        sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        if sigma.numel() != 1:
            raise ValueError(f"MiniMax-H3 currently requires one shared base sigma, got shape {tuple(sigma.shape)}.")
        return _shift_sigma(sigma, self.video_shift), _shift_sigma(sigma, self.audio_shift)

    def _prepare_condition_for_rollout(self, condition, broadcast):
        condition = dict(condition)
        fixed_frames = self.options.fixed_num_frames
        references = tuple(condition.get("references", ()))
        if fixed_frames is not None and references:
            if any(reference.kind != "image" for reference in references):
                raise ValueError("H3 fixed_num_frames can override only image-only reference caches.")
            condition["target_num_frames"] = fixed_frames
        clean = condition.get("condition_video_latents")
        if clean is None:
            return condition
        if references:
            shapes = tuple((reference.num_latent_frames, reference.latent_height, reference.latent_width) for reference in references if reference.kind != "audio")
        else:
            height = condition.get("target_height")
            width = condition.get("target_width")
            if height is None or width is None:
                raise ValueError("H3 keyframe conditions require cached target_height and target_width.")
            shapes = ((1, height // self.model.vae_spatial_scale_factor, width // self.model.vae_spatial_scale_factor),) * len(condition["keyframe_anchors"])
        noise = keyframe_condition_noise(
            shapes,
            self.model.patch_size,
            self.model.video_latent_channels,
            device=clean.device,
            dtype=torch.float32,
        )
        if noise.shape != clean.shape:
            raise ValueError(f"H3 cached visual rows {tuple(clean.shape)} do not match noise geometry {tuple(noise.shape)}.")
        condition["noised_condition_video_latents"] = broadcast(KEYFRAME_NOISE_AUG * clean.float() + (1.0 - KEYFRAME_NOISE_AUG) * noise)
        return condition

    def _layout(self, condition, shape):
        tags = condition["text_token_tags"]
        if tags.ndim != 1:
            raise ValueError(f"MiniMax-H3 text_token_tags must be one-dimensional, got {tuple(tags.shape)}.")
        references = tuple(condition.get("references", ()))
        anchors = tuple(condition.get("keyframe_anchors", ()))
        for name, expected in (
            ("target_num_frames", shape.num_frames),
            ("target_height", shape.latent_height * self.model.vae_spatial_scale_factor),
            ("target_width", shape.latent_width * self.model.vae_spatial_scale_factor),
        ):
            cached = condition.get(name)
            if cached is not None and int(cached) != expected:
                raise ValueError(f"H3 cached {name}={cached} differs from generated geometry {expected}.")
        key = (
            tuple(tags.detach().cpu().tolist()),
            references,
            anchors,
            shape.latent_frames,
            shape.latent_height,
            shape.latent_width,
            shape.audio_latents,
            self.model.patch_size,
            self.device,
        )
        layout = self._layout_cache.get(key)
        if layout is None:
            geometry = (
                shape.latent_frames,
                shape.latent_height,
                shape.latent_width,
                shape.audio_latents,
                self.model.patch_size,
            )
            if references:
                layout = build_ref2av_packed_sequence(tags.detach().cpu(), references, *geometry).to(self.device)
            else:
                layout = build_packed_sequence(tags.detach().cpu(), *geometry, keyframe_anchors=anchors).to(self.device)
            self._layout_cache[key] = layout
            while len(self._layout_cache) > self.options.layout_cache_size:
                self._layout_cache.popitem(last=False)
        else:
            self._layout_cache.move_to_end(key)
        return layout

    @staticmethod
    def _mix_noise(latent, noise, sigma):
        expanded = _expand_sigma(sigma, latent.ndim)
        return ((1.0 - expanded) * latent.float() + expanded * noise.float()).to(latent.dtype)

    @staticmethod
    def _cleanward_step(sample, velocity, sigma, sigma_next):
        current = _expand_sigma(sigma, sample.ndim)
        following = _expand_sigma(sigma_next, sample.ndim)
        return (sample.float() + (current - following) * velocity.float()).to(sample.dtype)

    @staticmethod
    def _cleanward_x0(sample, velocity, sigma):
        expanded = _expand_sigma(sigma, sample.ndim)
        return (sample.float() + expanded * velocity.float()).to(sample.dtype)

    @staticmethod
    def _validate_latents(latents):
        if not isinstance(latents, MiniMaxH3JointLatents):
            raise TypeError(f"Expected MiniMaxH3JointLatents, got {type(latents)!r}.")
        _require_singleton_tensor(latents.video, "MiniMax-H3 video latent")
        _require_singleton_tensor(latents.audio, "MiniMax-H3 audio latent")
