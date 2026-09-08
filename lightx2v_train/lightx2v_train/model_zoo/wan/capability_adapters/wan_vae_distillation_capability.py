"""Wan VAE component distillation on independently encoded RGB crops."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightx2v_train.model_capabilities import (
    BoundCapability,
    LossResult,
    VAEDistillationCapability,
    VAEDistillationStepContext,
)
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import (
    MiniMaxH3VAEDistillationCapability as _SharedLosses,
)


_LOSS_NAMES = ("reconstruction", "perceptual", "feature", "posterior", "auxiliary", "adversarial")


@dataclass(frozen=True)
class _Stage:
    name: str
    start_iter: int
    crop_num_frames: int
    crop_size: int
    weights: dict[str, float]


class WanVAEDistillationCapability(BoundCapability, VAEDistillationCapability):
    # Only geometry-independent losses are shared; Wan clips have their own causal origin.
    _charbonnier = staticmethod(_SharedLosses._charbonnier)
    _sample_frame_indices = staticmethod(_SharedLosses._sample_frame_indices)
    _perceptual_loss = _SharedLosses._perceptual_loss
    _spatial_gradient_loss = _SharedLosses._spatial_gradient_loss
    _spatiotemporal_loss = _SharedLosses._spatiotemporal_loss
    _auxiliary_loss = _SharedLosses._auxiliary_loss

    def __init__(self, model, config):
        super().__init__(model)
        self.crop_seed = int(config.get("crop_seed", 42))
        self.charbonnier_epsilon = float(config.get("charbonnier_epsilon", 1e-3))
        self.perceptual_network = config.get("perceptual_network", "vgg")
        self.perceptual_num_frames = int(config.get("perceptual_num_frames", 2))
        self.perceptual_frame_batch_size = int(config.get("perceptual_frame_batch_size", 2))
        self.perceptual_gradient_checkpointing = config.get("perceptual_gradient_checkpointing", True)
        self._perceptual_model = None
        if self.charbonnier_epsilon <= 0 or min(self.perceptual_num_frames, self.perceptual_frame_batch_size) < 1:
            raise ValueError("Charbonnier epsilon and perceptual frame counts must be positive.")

        self.stages = self._build_stages(config)
        self.uses_feature_loss = any(stage.weights["feature"] > 0 for stage in self.stages)
        auxiliary = config.get("auxiliary_decoder", {})
        default_anchors = (5, 7, 8) if model.component == "encoder" else (8, 11, 12)
        self.auxiliary_feature_indices = tuple(auxiliary.get("student_feature_indices", default_anchors))
        self.auxiliary_ramp_iters = int(auxiliary.get("ramp_iters", 200))
        self.auxiliary_weights = {
            "reconstruction": float(auxiliary.get("reconstruction_weight", 1.0)),
            "spatial_gradient": float(auxiliary.get("spatial_gradient_weight", 0.1)),
            "spatiotemporal": float(auxiliary.get("spatiotemporal_weight", 0.1)),
        }
        auxiliary_starts = [stage.start_iter for stage in self.stages if stage.weights["auxiliary"] > 0]
        self.uses_auxiliary_loss = bool(auxiliary_starts)
        self.auxiliary_start_iter = min(auxiliary_starts) if auxiliary_starts else 0
        adversarial_starts = [stage.start_iter for stage in self.stages if stage.weights["adversarial"] > 0]
        self.adversarial_start_iter = min(adversarial_starts) if adversarial_starts else 0
        if self.uses_auxiliary_loss:
            if not self.auxiliary_feature_indices or len(set(self.auxiliary_feature_indices)) != len(self.auxiliary_feature_indices):
                raise ValueError("Auxiliary feature indices must be nonempty and unique.")
            feature_count = 10 if model.component == "encoder" else 14
            if min(self.auxiliary_feature_indices) < 0 or max(self.auxiliary_feature_indices) >= feature_count:
                raise ValueError("Auxiliary index is outside the Wan component's residual blocks.")
            if self.auxiliary_ramp_iters < 0 or min(self.auxiliary_weights.values()) < 0 or sum(self.auxiliary_weights.values()) <= 0:
                raise ValueError("Auxiliary weights require a positive total; weights and ramp must be nonnegative.")

    def _build_stages(self, config):
        posterior = 1.0 if self.model.component == "encoder" else 0.0
        base = dict.fromkeys(_LOSS_NAMES, 0.0)
        base.update(reconstruction=1.0, posterior=posterior)
        base.update({name: float(config.get(f"{name}_weight", base[name])) for name in _LOSS_NAMES})
        stage_configs = config.get("stages")
        if stage_configs is None:
            phase = config.get("phase", "recover")
            if phase == "search":
                stage_configs = [{"name": "search", "start_iter": 0, "weights": {"perceptual": 0.05}}]
            elif phase == "recover":
                stage_configs = [
                    {"name": "recover_early", "start_iter": 0, "weights": {"perceptual": 0.1, "feature": 0.01}},
                    {
                        "name": "recover_late", "start_iter": 600, "crop_num_frames": 65, "crop_size": 384,
                        "weights": {"perceptual": 0.2, "feature": 0.005, "auxiliary": 0.1, "adversarial": 0.5},
                    },
                ]
            else:
                raise ValueError(f"Unknown Wan VAE distillation phase: {phase}")
        stages = []
        for entry in stage_configs:
            unknown = set(entry.get("weights", {})) - set(_LOSS_NAMES)
            if unknown:
                raise ValueError(f"Unknown Wan VAE loss weights: {sorted(unknown)}")
            weights = {**base, **{name: float(value) for name, value in entry.get("weights", {}).items()}}
            if min(weights.values()) < 0 or sum(weights.values()) <= 0:
                raise ValueError("Stage weights must be nonnegative with a positive total.")
            if self.model.component == "decoder" and weights["posterior"]:
                raise ValueError("Posterior alignment is an encoder-only loss.")
            frames = int(entry.get("crop_num_frames", config.get("crop_num_frames", 33)))
            size = int(entry.get("crop_size", config.get("crop_size", 256)))
            if frames < 1 or (frames - 1) % 4 or size < 8 or size % 8:
                raise ValueError("Wan training crops require 4n+1 frames and a positive multiple-of-8 size.")
            stages.append(_Stage(entry["name"], int(entry["start_iter"]), frames, size, weights))
        starts = [stage.start_iter for stage in stages]
        if not starts or starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError("Distillation stages must start at zero with strictly increasing start_iter values.")
        return tuple(stages)

    def _stage_for(self, iteration):
        index = max(index for index, stage in enumerate(self.stages) if stage.start_iter <= iteration)
        return index, self.stages[index]

    def _crop_video(self, video, stage, generator):
        frames, size = stage.crop_num_frames, stage.crop_size
        _, _, total_frames, height, width = video.shape
        if total_frames < frames or min(height, width) < size:
            raise ValueError(f"Video {tuple(video.shape)} is smaller than training crop ({frames}, {size}, {size}).")
        crops = []
        for item in video:
            starts = [int(torch.randint(length - count + 1, (), generator=generator)) for length, count in (
                (total_frames, frames), (height, size), (width, size),
            )]
            t, y, x = starts
            crops.append(item[:, t : t + frames, y : y + size, x : x + size])
        return torch.stack(crops)

    def _auxiliary_ramp(self, iteration):
        if not self.uses_auxiliary_loss or iteration < self.auxiliary_start_iter:
            return 0.0
        return min(1.0, (iteration - self.auxiliary_start_iter + 1) / max(1, self.auxiliary_ramp_iters))

    @staticmethod
    def _matching_mse(student, teacher):
        if student.shape != teacher.shape:
            raise ValueError(f"Distillation shapes differ: student={student.shape}, teacher={teacher.shape}")
        return F.mse_loss(student.float(), teacher.detach().float())

    def compute_loss(self, batch, context: VAEDistillationStepContext) -> LossResult:
        stage_index, stage = self._stage_for(context.iteration)
        weights = dict(stage.weights)
        auxiliary_ramp = self._auxiliary_ramp(context.iteration) if weights["auxiliary"] else 0.0
        weights["auxiliary"] *= auxiliary_ramp
        generator = torch.Generator().manual_seed(self.crop_seed + context.iteration * 1_000_003 + context.micro_step)
        video = self._crop_video(batch["inputs"]["video"], stage, generator)
        video = video.to(device=self.model.device, dtype=torch.float32)
        auxiliary_index = None
        if weights["auxiliary"]:
            choice = int(torch.randint(len(self.auxiliary_feature_indices), (), generator=generator))
            auxiliary_index = self.auxiliary_feature_indices[choice]
        if weights["adversarial"] and context.adversarial_objective is None:
            raise RuntimeError("The active Wan VAE stage requires training.vae_distillation.gan.enabled=true.")

        # Re-encode this crop from time zero; cached full-video latents have a different causal history.
        outputs = self.model.distillation_forward(
            video, running_dtype=context.running_dtype, return_features=bool(weights["feature"]),
            auxiliary_feature_index=auxiliary_index,
        )
        prediction = outputs["prediction"].float()
        if prediction.shape != video.shape:
            raise ValueError(f"Wan reconstruction shape {prediction.shape} differs from crop {video.shape}.")
        zero = prediction.new_zeros(())
        losses = dict.fromkeys(_LOSS_NAMES, zero)
        metrics = {}
        if weights["reconstruction"]:
            losses["reconstruction"] = self._charbonnier(prediction, video * 2 - 1, self.charbonnier_epsilon)
        prediction_rgb = (prediction + 1).mul(0.5).clamp(0, 1)
        if weights["perceptual"]:
            losses["perceptual"] = self._perceptual_loss(prediction_rgb, video)
        if weights["feature"]:
            feature_losses = [self._matching_mse(student, teacher) for student, teacher in zip(
                outputs["student_features"], outputs["teacher_features"], strict=True,
            )]
            if not feature_losses:
                raise ValueError("Feature loss requires matching student and teacher anchors.")
            losses["feature"] = torch.stack(feature_losses).mean()
        if weights["posterior"]:
            mean_loss = self._matching_mse(outputs["student_mu"], outputs["teacher_mu"])
            std_loss = self._matching_mse(outputs["student_std"], outputs["teacher_std"])
            losses["posterior"] = mean_loss + std_loss
            metrics.update(posterior_mean=mean_loss, posterior_std=std_loss)
        if weights["auxiliary"]:
            auxiliary = outputs["auxiliary_prediction"]
            teacher = outputs["teacher_prediction"]
            if auxiliary.shape != teacher.shape or auxiliary.shape != prediction.shape:
                raise ValueError("Auxiliary, teacher, and student reconstructions must cover the same RGB crop.")
            losses["auxiliary"], auxiliary_metrics = self._auxiliary_loss(auxiliary, teacher)
            metrics.update(auxiliary_metrics)
        if weights["adversarial"]:
            latents = outputs["teacher_latents"].detach()
            # Wan's first latent represents one frame; each subsequent latent represents four.
            condition = torch.cat((latents[:, :, :1], latents[:, :, 1:].repeat_interleave(4, dim=2)), dim=2)
            if condition.shape[2] != prediction.shape[2]:
                raise ValueError("Wan GAN condition must cover the same temporal crop as the decoded RGB.")
            losses["adversarial"], adversarial_metrics = context.adversarial_objective(
                prediction_rgb, video, condition,
                context.iteration - self.adversarial_start_iter,
            )
            metrics.update(adversarial_metrics)

        loss = sum(weights[name] * losses[name] for name in _LOSS_NAMES)
        metrics.update(losses)
        metrics.update({f"weighted_{name}": weights[name] * losses[name] for name in _LOSS_NAMES})
        metrics.update(
            stage_index=float(stage_index), auxiliary_ramp=auxiliary_ramp,
            auxiliary_feature_index=float(auxiliary_index) if auxiliary_index is not None else -1.0,
            supervised_frames=float(video.shape[2]), supervised_height=float(video.shape[3]),
            supervised_width=float(video.shape[4]),
        )
        return LossResult(loss=loss, metrics=metrics)
