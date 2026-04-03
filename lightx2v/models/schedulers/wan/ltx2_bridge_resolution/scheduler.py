import torch
from loguru import logger

from lightx2v.models.schedulers.wan.changing_resolution.scheduler import (
    WanScheduler4ChangingResolution,
)
from lightx2v_platform.base.global_var import AI_DEVICE


class WanScheduler4LTX2BridgeInterface:
    """独立的 changing_resolution scheduler 入口。

    这条链路不改原有 changing_resolution 文件，而是通过新的 mixin
    复用其 prepare_latents / step_post / add_noise 等基础逻辑，
    只把“近似 x0 迁移到下一阶段分辨率”的实现替换为可插拔 bridge。
    """

    def __new__(cls, father_scheduler, config):
        class NewClass(WanScheduler4LTX2Bridge, father_scheduler):
            def __init__(self, config):
                father_scheduler.__init__(self, config)
                WanScheduler4LTX2Bridge.__init__(self, config)

        return NewClass(config)


class WanScheduler4LTX2Bridge(WanScheduler4ChangingResolution):
    def __init__(self, config):
        super().__init__(config)
        self.clean_latent_resizer = None

    def set_clean_latent_resizer(self, resizer):
        self.clean_latent_resizer = resizer

    def _resize_clean_latent_to_next_stage(self, denoised_sample, target_latent_shape):
        denoised_sample_5d = denoised_sample.unsqueeze(0)
        target_size = target_latent_shape[1:]

        # 只在“空间 x2 升分”时走 LTX2 bridge，其余情况继续回退到原始插值逻辑。
        can_use_bridge = (
            self.clean_latent_resizer is not None
            and target_latent_shape[1] == denoised_sample.shape[1]
            and target_latent_shape[2] == denoised_sample.shape[2] * 2
            and target_latent_shape[3] == denoised_sample.shape[3] * 2
        )
        if can_use_bridge:
            logger.info(
                "Use LTX2 bridge to resize WAN clean latent: "
                f"{tuple(denoised_sample.shape)} -> {tuple(target_latent_shape)}"
            )
            return self.clean_latent_resizer.resize(
                latent=denoised_sample,
                target_latent_shape=target_latent_shape,
                step_index=self.step_index,
                changing_resolution_index=self.changing_resolution_index,
            )

        return torch.nn.functional.interpolate(
            denoised_sample_5d,
            size=target_size,
            mode="trilinear",
        ).squeeze(0)

    def step_post_upsample(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        sigma_t = self.sigmas[self.step_index]
        x0_pred = sample - sigma_t * model_output
        denoised_sample = x0_pred.to(sample.dtype)

        target_latent_shape = self.latents_list[self.changing_resolution_index + 1].shape
        clean_sample = self._resize_clean_latent_to_next_stage(denoised_sample, target_latent_shape)

        noisy_sample = self.add_noise(
            clean_sample,
            self.latents_list[self.changing_resolution_index + 1],
            self.timesteps[self.step_index + 1],
        )

        self.latents = noisy_sample
        self.set_timesteps(
            self.infer_steps,
            device=AI_DEVICE,
            shift=self.sample_shift + self.changing_resolution_index + 1,
        )
