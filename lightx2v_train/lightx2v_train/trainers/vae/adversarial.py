"""Adversarial objective owned by the VAE distillation trainer."""

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.native.minimax_h3.vae_adversarial import (
    DiscriminatorLoss,
    LatentConditionedFramePatchDiscriminator,
    LatentConditionedVideoPatchDiscriminator,
    discriminator_lsgan_loss,
    make_ensemble_seraena_correction_target,
)
from lightx2v_train.runtime.ddp import unwrap_ddp_module
from lightx2v_train.runtime.distributed import get_data_parallel_group, is_distributed


class VAEAdversarialObjective:
    def __init__(self, config, *, device, latent_channels, gradient_accumulation_iters):
        self.gradient_accumulation_iters = gradient_accumulation_iters
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.spatial_compression_ratio = int(config.get("spatial_compression_ratio", 16))
        if self.spatial_compression_ratio < 1:
            raise ValueError("GAN spatial_compression_ratio must be positive.")
        crop_specs = config.get("crop_specs")
        if crop_specs is None:
            crop_specs = [
                {
                    "crop_size": config.get("crop_size", 128),
                    "max_frames": config.get("max_frames", 5),
                    "weight": 1.0,
                }
            ]
        self.crop_specs = tuple(
            (
                int(spec["crop_size"]),
                int(spec["max_frames"]),
                float(spec.get("weight", 1.0)),
            )
            for spec in crop_specs
        )
        if any(size < 1 or frames < 1 or weight < 0 for size, frames, weight in self.crop_specs):
            raise ValueError("GAN crop sizes and frame counts must be positive, with non-negative weights.")
        if sum(weight for _, _, weight in self.crop_specs) <= 0:
            raise ValueError("At least one GAN crop spec must have positive weight.")
        self.correction_scale = float(config.get("correction_scale", 0.05))
        self.correction_clamp = float(config.get("correction_clamp", 4.0))
        self.discriminator_warmup_iters = int(config.get("discriminator_warmup_iters", 1000))
        self.generator_ramp_iters = int(config.get("generator_ramp_iters", 5000))
        self.discriminator_steps = 0
        self._has_gradients = False

        discriminator_config = config.get("discriminator", {})
        frame_discriminator_config = config.get("frame_discriminator", {})
        frame_discriminator_enabled = bool(frame_discriminator_config.get("enabled", False))
        self.frame_discriminator_weight = (
            float(frame_discriminator_config.get("weight", 1.0))
            if frame_discriminator_enabled
            else 0.0
        )
        if frame_discriminator_enabled and self.frame_discriminator_weight <= 0:
            raise ValueError("The enabled frame discriminator requires a positive weight.")

        seed = int(config.get("seed", 42))
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            discriminator = LatentConditionedVideoPatchDiscriminator(
                latent_channels=latent_channels,
                condition_channels=int(discriminator_config.get("condition_channels", 8)),
                base_channels=int(discriminator_config.get("base_channels", 32)),
                channel_multipliers=tuple(discriminator_config.get("channel_multipliers", (1, 2, 4, 8))),
                temporal_strides=tuple(discriminator_config.get("temporal_strides", (1, 1, 2, 2))),
                group_norm_groups=int(discriminator_config.get("group_norm_groups", 16)),
                spectral_normalization=bool(discriminator_config.get("spectral_normalization", True)),
                gradient_checkpointing=bool(discriminator_config.get("gradient_checkpointing", False)),
            ).to(device=device, dtype=torch.float32)
            frame_discriminator = None
            if frame_discriminator_enabled:
                frame_discriminator = LatentConditionedFramePatchDiscriminator(
                    latent_channels=latent_channels,
                    condition_channels=int(frame_discriminator_config.get("condition_channels", 16)),
                    base_channels=int(frame_discriminator_config.get("base_channels", 64)),
                    channel_multipliers=tuple(
                        frame_discriminator_config.get("channel_multipliers", (1, 2, 4, 8, 8, 8))
                    ),
                    spatial_strides=tuple(
                        frame_discriminator_config.get("spatial_strides", (2, 2, 2, 2, 1, 1))
                    ),
                    group_norm_groups=int(frame_discriminator_config.get("group_norm_groups", 32)),
                    spectral_normalization=bool(
                        frame_discriminator_config.get("spectral_normalization", True)
                    ),
                    gradient_checkpointing=bool(frame_discriminator_config.get("gradient_checkpointing", False)),
                ).to(device=device, dtype=torch.float32)

        if is_distributed():
            ddp_kwargs = {
                "process_group": get_data_parallel_group(),
                "broadcast_buffers": True,
                "find_unused_parameters": False,
            }
            if device.type == "cuda":
                ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
            discriminator = DistributedDataParallel(discriminator, **ddp_kwargs)
            if frame_discriminator is not None:
                frame_discriminator = DistributedDataParallel(frame_discriminator, **ddp_kwargs)
        self.discriminator = discriminator
        self.frame_discriminator = frame_discriminator
        self.discriminator_parameters = list(self.discriminator.parameters())
        if self.frame_discriminator is not None:
            self.discriminator_parameters.extend(self.frame_discriminator.parameters())

        optimizer_config = config.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.discriminator_parameters,
            lr=float(optimizer_config.get("learning_rate", 1e-4)),
            betas=(
                float(optimizer_config.get("adam_beta1", 0.5)),
                float(optimizer_config.get("adam_beta2", 0.9)),
            ),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
            eps=float(optimizer_config.get("adam_epsilon", 1e-8)),
        )
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def module(self):
        return unwrap_ddp_module(self.discriminator)

    @property
    def frame_module(self):
        return unwrap_ddp_module(self.frame_discriminator)

    def set_gradient_sync(self, enabled):
        if isinstance(self.discriminator, DistributedDataParallel):
            self.discriminator.require_backward_grad_sync = enabled
        if isinstance(self.frame_discriminator, DistributedDataParallel):
            self.frame_discriminator.require_backward_grad_sync = enabled

    @staticmethod
    def _random_start(length, crop_length, *, device):
        if crop_length >= length:
            return 0
        return int(torch.randint(length - crop_length + 1, (), device=device).item())

    def _sample_crop_spec(self, device):
        weights = torch.tensor(
            [spec[2] for spec in self.crop_specs],
            dtype=torch.float32,
            device=device,
        )
        index = int(torch.multinomial(weights, 1).item())
        return self.crop_specs[index][:2]

    def _aligned_crop(self, fake_video, real_video, latent, crop_size, max_frames):
        # The default preserves existing H3 crops; Wan2.1 uses an 8x spatial ratio.
        ratio = self.spatial_compression_ratio
        latent_crop_height = min(latent.shape[-2], max(1, crop_size // ratio))
        latent_crop_width = min(latent.shape[-1], max(1, crop_size // ratio))
        latent_y = self._random_start(latent.shape[-2], latent_crop_height, device=fake_video.device)
        latent_x = self._random_start(latent.shape[-1], latent_crop_width, device=fake_video.device)

        scale_y = fake_video.shape[-2] // latent.shape[-2]
        scale_x = fake_video.shape[-1] // latent.shape[-1]
        pixel_y = latent_y * scale_y
        pixel_x = latent_x * scale_x
        pixel_height = latent_crop_height * scale_y
        pixel_width = latent_crop_width * scale_x

        frame_count = min(fake_video.shape[2], max_frames)
        frame_start = self._random_start(fake_video.shape[2], frame_count, device=fake_video.device)
        if latent.shape[2] == fake_video.shape[2]:
            latent_start = frame_start
            latent_end = frame_start + frame_count
        else:
            latent_start = frame_start * latent.shape[2] // fake_video.shape[2]
            latent_end = max(
                latent_start + 1,
                (frame_start + frame_count) * latent.shape[2] // fake_video.shape[2],
            )
            latent_end = min(latent.shape[2], latent_end)

        video_slice = (
            slice(None),
            slice(None),
            slice(frame_start, frame_start + frame_count),
            slice(pixel_y, pixel_y + pixel_height),
            slice(pixel_x, pixel_x + pixel_width),
        )
        latent_slice = (
            slice(None),
            slice(None),
            slice(latent_start, latent_end),
            slice(latent_y, latent_y + latent_crop_height),
            slice(latent_x, latent_x + latent_crop_width),
        )
        return (
            fake_video[video_slice],
            real_video[video_slice],
            latent[latent_slice],
            (frame_count, pixel_height, pixel_width),
        )

    def __call__(self, fake_video, real_video, latent, stage_step):
        crop_size, max_frames = self._sample_crop_spec(fake_video.device)
        fake_video, real_video, latent, crop_shape = self._aligned_crop(
            fake_video,
            real_video,
            latent,
            crop_size,
            max_frames,
        )
        fake_video = fake_video.float() * 2.0 - 1.0
        real_video = real_video.float() * 2.0 - 1.0
        latent = latent.float()

        batch_size = fake_video.shape[0]
        joint_video = torch.cat((real_video, fake_video.detach()), dim=0)
        joint_latent = torch.cat((latent, latent), dim=0)
        self.discriminator.train()
        joint_logits = self.discriminator(joint_video, joint_latent)
        real_logits = tuple(value[:batch_size] for value in joint_logits)
        fake_logits = tuple(value[batch_size:] for value in joint_logits)
        discriminator_3d = discriminator_lsgan_loss(real_logits, fake_logits)
        (discriminator_3d.total / self.gradient_accumulation_iters).backward()

        zero = discriminator_3d.total.new_zeros(())
        discriminator_2d = DiscriminatorLoss(total=zero, real=zero, fake=zero)
        if self.frame_discriminator is not None:
            self.frame_discriminator.train()
            frame_logits = self.frame_discriminator(joint_video, joint_latent)
            frame_real_logits = tuple(value[:batch_size] for value in frame_logits)
            frame_fake_logits = tuple(value[batch_size:] for value in frame_logits)
            discriminator_2d = discriminator_lsgan_loss(frame_real_logits, frame_fake_logits)
            (
                self.frame_discriminator_weight
                * discriminator_2d.total
                / self.gradient_accumulation_iters
            ).backward()

        discriminator_loss = DiscriminatorLoss(
            total=discriminator_3d.total + self.frame_discriminator_weight * discriminator_2d.total,
            real=discriminator_3d.real + self.frame_discriminator_weight * discriminator_2d.real,
            fake=discriminator_3d.fake + self.frame_discriminator_weight * discriminator_2d.fake,
        )
        self._has_gradients = True

        ramp_step = stage_step - self.discriminator_warmup_iters
        if ramp_step < 0:
            generator_loss = fake_video.sum() * 0.0
            correction_rms = fake_video.new_zeros(())
            feature_distance = fake_video.new_zeros(())
            ramp = 0.0
        else:
            ramp = min(1.0, (ramp_step + 1) / max(1, self.generator_ramp_iters))
            branches = [(self.module, 1.0)]
            if self.frame_discriminator is not None:
                branches.append((self.frame_module, self.frame_discriminator_weight))
            training_states = [discriminator.training for discriminator, _ in branches]
            for discriminator, _ in branches:
                discriminator.eval()
            try:
                correction = make_ensemble_seraena_correction_target(
                    tuple(branches),
                    real_video,
                    fake_video,
                    latent,
                    correction_scale=self.correction_scale * ramp,
                    normalized_clamp=self.correction_clamp,
                )
            finally:
                for (discriminator, _), training in zip(branches, training_states, strict=True):
                    discriminator.train(training)
            generator_loss = F.mse_loss(fake_video.float(), correction.target.float())
            correction_rms = correction.correction.float().square().mean().sqrt()
            feature_distance = correction.feature_distance.mean()

        feature_distance_3d = feature_distance
        feature_distance_2d = zero
        if ramp_step >= 0:
            feature_distance_3d = correction.branch_feature_distances[0].mean()
            if self.frame_discriminator is not None:
                feature_distance_2d = correction.branch_feature_distances[1].mean()

        return generator_loss, {
            "discriminator": discriminator_loss.total,
            "discriminator_real": discriminator_loss.real,
            "discriminator_fake": discriminator_loss.fake,
            "discriminator_3d": discriminator_3d.total,
            "discriminator_real_3d": discriminator_3d.real,
            "discriminator_fake_3d": discriminator_3d.fake,
            "discriminator_2d": discriminator_2d.total,
            "discriminator_real_2d": discriminator_2d.real,
            "discriminator_fake_2d": discriminator_2d.fake,
            "adversarial_feature_distance": feature_distance,
            "adversarial_feature_distance_3d": feature_distance_3d,
            "adversarial_feature_distance_2d": feature_distance_2d,
            "adversarial_correction_rms": correction_rms,
            "adversarial_ramp": ramp,
            "adversarial_crop_frames": float(crop_shape[0]),
            "adversarial_crop_height": float(crop_shape[1]),
            "adversarial_crop_width": float(crop_shape[2]),
        }

    def step(self):
        if not self._has_gradients:
            return
        torch.nn.utils.clip_grad_norm_(self.discriminator_parameters, self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.discriminator_steps += 1
        self._has_gradients = False

    def state_dict(self):
        state = {
            "discriminator": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "discriminator_steps": self.discriminator_steps,
        }
        if self.frame_discriminator is not None:
            state["frame_discriminator"] = self.frame_module.state_dict()
        return state

    def load_state_dict(self, state):
        self.module.load_state_dict(state["discriminator"], strict=True)
        if self.frame_discriminator is not None:
            self.frame_module.load_state_dict(state["frame_discriminator"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.discriminator_steps = int(state["discriminator_steps"])
