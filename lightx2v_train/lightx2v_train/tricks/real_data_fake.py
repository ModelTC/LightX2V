from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from lightx2v_train.trainers.dmd.math import (
    euler_step,
    velocity_to_x0,
)
from lightx2v_train.trainers.phased_dmd.math import (
    phased_coefficients,
    phased_forward,
    phased_velocity_target,
)

from .base import TrainerTrick, TrickLossResult


@dataclass(frozen=True)
class RealDataFakeRegionConfig:
    role: str
    enabled: bool = False
    weight: float = 0.1
    timestep_list: tuple[int, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        role: str,
        config: Mapping[str, Any],
    ) -> "RealDataFakeRegionConfig":
        if not isinstance(config, Mapping):
            raise ValueError(f"Real-data fake role {role!r} must be a mapping.")
        return cls(
            role=role,
            enabled=bool(config.get("enabled", False)),
            weight=float(config.get("weight", 0.1)),
            timestep_list=tuple(int(timestep) for timestep in config.get("timestep_list", ())),
        )


@dataclass(frozen=True)
class RealDataFakeConfig:
    regions: Mapping[str, RealDataFakeRegionConfig]


@dataclass(frozen=True)
class RealDataFakeSetupContext:
    mode: str
    scheduler: Any
    denoising_scheduler: Any
    num_train_timestep: int
    warp_denoising_step: bool
    score_timestep_shift: float = 1.0
    min_step: int = 0
    max_step: int = 1000
    min_score_timestep: int = 0
    match_timestep: int = 0
    score_timestep_margin: int = 0
    score_timestep_min: int = 1
    score_timestep_max: int = 1000
    phased_eps: float = 1.0e-8


@dataclass(frozen=True)
class RealDataFakeStepContext:
    region: str
    fake_model: Any
    sample: Mapping[str, Any]
    condition: Any
    negative_condition: Any
    device: torch.device
    dtype: torch.dtype
    extract_real_latents: Callable[[Mapping[str, Any]], torch.Tensor]
    sample_synced_int: Callable[[int, int], int]
    broadcast_noise: Callable[[torch.Tensor], torch.Tensor]
    predict_student_velocity: Callable[
        [str, torch.Tensor, torch.Tensor, Any, bool],
        torch.Tensor,
    ]
    predict_fake_velocity: Callable[
        [Any, torch.Tensor, torch.Tensor, Any],
        torch.Tensor,
    ]
    predict_teacher_velocity: Callable[
        [str, torch.Tensor, torch.Tensor, Any, Any],
        torch.Tensor,
    ]
    dmd_loss: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]


class RealDataFakeTrick(
    TrainerTrick[
        RealDataFakeSetupContext,
        RealDataFakeStepContext,
        RealDataFakeStepContext,
        None,
    ]
):
    """Train auxiliary fake-score models from real-data student samples."""

    name = "real_data_fake"

    def __init__(self, config: RealDataFakeConfig):
        super().__init__(enabled=any(region.enabled for region in config.regions.values()))
        self.config = config
        self._setup_context: RealDataFakeSetupContext | None = None
        self.validate()

    @classmethod
    def from_mappings(
        cls,
        configs: Mapping[
            str,
            tuple[str, Mapping[str, Any]],
        ],
    ) -> "RealDataFakeTrick":
        if not isinstance(configs, Mapping):
            raise ValueError("Real-data fake configuration must be a mapping.")
        return cls(
            RealDataFakeConfig(
                regions={
                    region: RealDataFakeRegionConfig.from_mapping(
                        role=role,
                        config=config,
                    )
                    for region, (role, config) in configs.items()
                }
            )
        )

    def validate(self) -> None:
        for region, config in self.config.regions.items():
            if config.weight < 0:
                raise ValueError(f"Real-data fake region {region!r} weight must be non-negative.")
            if config.enabled and not config.timestep_list:
                raise ValueError(f"Real-data fake region {region!r} requires a non-empty timestep_list.")
            if any(timestep < 0 for timestep in config.timestep_list):
                raise ValueError(f"Real-data fake region {region!r} timesteps must be non-negative.")

    def setup(self, context: RealDataFakeSetupContext) -> None:
        if context.mode not in {"standard", "phased"}:
            raise ValueError("Real-data fake mode must be 'standard' or 'phased'.")
        if context.num_train_timestep < 1:
            raise ValueError("Real-data fake num_train_timestep must be positive.")
        if context.mode == "phased":
            if not 0 < context.match_timestep <= context.num_train_timestep:
                raise ValueError("Phased real-data fake match_timestep is out of range.")
            if context.phased_eps <= 0:
                raise ValueError("Phased real-data fake epsilon must be positive.")
        self.validate_schedule(
            mode=context.mode,
            num_train_timestep=context.num_train_timestep,
            match_timestep=context.match_timestep,
        )
        self._setup_context = context

    def validate_schedule(
        self,
        *,
        mode: str,
        num_train_timestep: int,
        match_timestep: int = 0,
    ) -> None:
        for region, config in self.config.regions.items():
            if not config.enabled:
                continue
            if any(timestep < 1 or timestep > num_train_timestep for timestep in config.timestep_list):
                raise ValueError(f"Real-data fake region {region!r} timestep_list values must be in [1, {num_train_timestep}].")
            if mode == "phased" and region == "high" and any(timestep <= match_timestep for timestep in config.timestep_list):
                raise ValueError("High real-data fake timestep_list must be above match_timestep.")
            if mode == "phased" and region == "low" and any(timestep > match_timestep for timestep in config.timestep_list):
                raise ValueError("Low real-data fake timestep_list must not exceed match_timestep.")

    @property
    def setup_context(self) -> RealDataFakeSetupContext:
        if self._setup_context is None:
            raise RuntimeError("The real-data fake trick is not initialized.")
        return self._setup_context

    def enabled_for(self, region: str) -> bool:
        config = self.config.regions.get(region)
        return bool(config is not None and config.enabled)

    def role_for(self, region: str) -> str:
        return self._region_config(region).role

    def _region_config(self, region: str) -> RealDataFakeRegionConfig:
        if region not in self.config.regions:
            available = ", ".join(sorted(self.config.regions))
            raise ValueError(f"Unknown real-data fake region {region!r}. Available regions: {available}.")
        return self.config.regions[region]

    def raw_timesteps_to_sigmas(
        self,
        raw_timesteps: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        setup = self.setup_context
        raw_timesteps = raw_timesteps.to(
            device=raw_timesteps.device,
            dtype=torch.long,
        )
        if setup.warp_denoising_step:
            timesteps = setup.denoising_scheduler.timesteps.to(
                device=raw_timesteps.device,
                dtype=torch.float32,
            )
            indices = setup.denoising_scheduler.num_train_timesteps - raw_timesteps
            warped_timesteps = timesteps[indices]
        else:
            warped_timesteps = raw_timesteps.float()
        return (warped_timesteps / setup.num_train_timestep).to(dtype=dtype)

    def _expand_to_ndim(
        self,
        value: torch.Tensor,
        ndim: int,
    ) -> torch.Tensor:
        return self.setup_context.scheduler._expand_to_ndim(value, ndim)

    def _euler_step(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        return euler_step(
            sample,
            velocity,
            sigma,
            sigma_next,
        )

    def _phase_sigma(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raw_timestep = torch.full(
            (int(batch_size),),
            self.setup_context.match_timestep,
            device=device,
            dtype=torch.long,
        )
        return self.raw_timesteps_to_sigmas(raw_timestep, dtype)

    def _build_student_outputs(
        self,
        region: str,
        xt: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        setup = self.setup_context
        zero_sigma = torch.zeros_like(sigma)
        student_x0 = self._euler_step(
            xt,
            velocity,
            sigma,
            zero_sigma,
        )
        if setup.mode == "phased" and region == "high":
            sigma_s = self._phase_sigma(
                xt.shape[0],
                xt.device,
                xt.dtype,
            )
            anchor = self._euler_step(
                xt,
                velocity,
                sigma,
                sigma_s,
            )
        else:
            sigma_s = zero_sigma
            anchor = student_x0
        return (
            anchor.to(dtype=xt.dtype),
            student_x0.to(dtype=xt.dtype),
            sigma_s,
        )

    def _sample_score_sigma(
        self,
        context: RealDataFakeStepContext,
        batch_size: int,
    ) -> torch.Tensor:
        setup = self.setup_context
        if setup.mode == "standard":
            min_timestep = max(0, int(setup.min_score_timestep))
            max_timestep = setup.num_train_timestep
            if max_timestep <= min_timestep:
                max_timestep = min(
                    setup.num_train_timestep,
                    min_timestep + 1,
                )
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                (int(batch_size),),
                device=context.device,
                dtype=torch.long,
            ).float()
            if setup.score_timestep_shift > 1:
                normalized = timestep / setup.num_train_timestep
                timestep = setup.score_timestep_shift * normalized / (1 + (setup.score_timestep_shift - 1) * normalized) * setup.num_train_timestep
            timestep = timestep.clamp(setup.min_step, setup.max_step)
            sigma = timestep / setup.num_train_timestep
        else:
            if context.region == "high":
                raw_min = setup.match_timestep + setup.score_timestep_margin
                raw_max = setup.num_train_timestep
            else:
                raw_min = 1
                raw_max = setup.match_timestep
            raw_min = max(1, int(raw_min), setup.score_timestep_min)
            raw_max = min(
                setup.num_train_timestep,
                int(raw_max),
                setup.score_timestep_max + 1,
            )
            raw_candidates = torch.arange(
                raw_min,
                raw_max,
                device=context.device,
                dtype=torch.long,
            )
            candidate_sigmas = self.raw_timesteps_to_sigmas(
                raw_candidates,
                dtype=torch.float32,
            )
            if candidate_sigmas.numel() == 0:
                raise RuntimeError(f"No valid real-data fake score timesteps remain in raw range [{raw_min}, {raw_max}).")
            candidate_indices = torch.randint(
                0,
                candidate_sigmas.numel(),
                (int(batch_size),),
                device=context.device,
                dtype=torch.long,
            )
            sigma = candidate_sigmas[candidate_indices]
        return context.broadcast_noise(sigma.to(dtype=context.dtype))

    def _phased_coefficients(
        self,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        ndim: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return phased_coefficients(
            sigma_s,
            sigma_t,
            ndim,
            self.setup_context.phased_eps,
        )

    def _score_forward(
        self,
        anchor: torch.Tensor,
        noise: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        if self.setup_context.mode == "standard":
            return self.setup_context.scheduler.add_noise(
                anchor,
                noise,
                sigma_t,
            )
        return phased_forward(
            anchor,
            noise,
            sigma_s,
            sigma_t,
            self.setup_context.phased_eps,
        )

    def _score_velocity_target(
        self,
        anchor: torch.Tensor,
        noise: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        if self.setup_context.mode == "standard":
            return noise - anchor.float()
        return phased_velocity_target(
            anchor,
            noise,
            sigma_s,
            sigma_t,
            self.setup_context.phased_eps,
        )

    def _velocity_to_x0(
        self,
        xt: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return velocity_to_x0(xt, velocity, sigma)

    def _real_state(
        self,
        context: RealDataFakeStepContext,
        *,
        grad_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self._region_config(context.region)
        real_latents = context.extract_real_latents(context.sample)
        timestep_index = context.sample_synced_int(
            0,
            len(config.timestep_list),
        )
        raw_timestep = torch.full(
            (real_latents.shape[0],),
            config.timestep_list[timestep_index],
            device=context.device,
            dtype=torch.long,
        )
        sigma = self.raw_timesteps_to_sigmas(
            raw_timestep,
            context.dtype,
        )
        noise = context.broadcast_noise(torch.randn_like(real_latents, dtype=torch.float32))
        with torch.no_grad():
            xt = self.setup_context.scheduler.add_noise(
                real_latents,
                noise,
                sigma,
            )
        velocity = context.predict_student_velocity(
            context.region,
            xt,
            sigma,
            context.condition,
            grad_enabled,
        )
        return self._build_student_outputs(
            context.region,
            xt,
            velocity,
            sigma,
        )

    def student_loss(
        self,
        context: RealDataFakeStepContext,
    ) -> TrickLossResult:
        config = self._region_config(context.region)
        if not config.enabled:
            raise RuntimeError(f"Real-data fake region {context.region!r} is disabled.")
        anchor, student_x0, sigma_s = self._real_state(
            context,
            grad_enabled=True,
        )
        sigma_t = self._sample_score_sigma(context, anchor.shape[0])
        noise = context.broadcast_noise(torch.randn_like(anchor, dtype=torch.float32))
        with torch.no_grad():
            score_xt = self._score_forward(
                anchor.detach(),
                noise,
                sigma_s,
                sigma_t,
            )
            context.fake_model.transformer.eval()
            velocity_fake = context.predict_fake_velocity(
                context.fake_model,
                score_xt,
                sigma_t,
                context.condition,
            )
            velocity_teacher = context.predict_teacher_velocity(
                context.region,
                score_xt,
                sigma_t,
                context.condition,
                context.negative_condition,
            )
            fake_x0 = self._velocity_to_x0(
                score_xt,
                velocity_fake,
                sigma_t,
            )
            teacher_x0 = self._velocity_to_x0(
                score_xt,
                velocity_teacher,
                sigma_t,
            )
        raw_loss = context.dmd_loss(
            student_x0,
            fake_x0,
            teacher_x0,
        )
        return TrickLossResult(
            loss=config.weight * raw_loss,
            metrics={"real_dmd": raw_loss.detach()},
        )

    def fake_loss(
        self,
        context: RealDataFakeStepContext,
    ) -> TrickLossResult:
        config = self._region_config(context.region)
        if not config.enabled:
            raise RuntimeError(f"Real-data fake region {context.region!r} is disabled.")
        anchor, _, sigma_s = self._real_state(
            context,
            grad_enabled=False,
        )
        sigma_t = self._sample_score_sigma(context, anchor.shape[0])
        noise = context.broadcast_noise(torch.randn_like(anchor, dtype=torch.float32))
        with torch.no_grad():
            score_xt = self._score_forward(
                anchor.detach(),
                noise,
                sigma_s,
                sigma_t,
            )
            velocity_target = self._score_velocity_target(
                anchor.detach(),
                noise,
                sigma_s,
                sigma_t,
            )
        context.fake_model.transformer.train()
        velocity_fake = context.predict_fake_velocity(
            context.fake_model,
            score_xt,
            sigma_t,
            context.condition,
        )
        raw_loss = F.mse_loss(
            velocity_fake.float(),
            velocity_target.float(),
            reduction="mean",
        )
        return TrickLossResult(
            loss=raw_loss,
            metrics={"fake_real": raw_loss.detach()},
        )

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {}
        for config in self.config.regions.values():
            metadata[f"{config.role}_enabled"] = config.enabled
            metadata[f"{config.role}_weight"] = config.weight
            metadata[f"{config.role}_timestep_list"] = list(config.timestep_list)
        return metadata
