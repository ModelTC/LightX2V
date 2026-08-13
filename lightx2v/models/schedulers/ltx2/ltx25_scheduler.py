import torch

from lightx2v.models.schedulers.ltx2.scheduler import LTX2Scheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class LTX25Scheduler(LTX2Scheduler):
    """LTX-2.5 scheduler with distilled stage-1 ancestral Euler sampling.

    The stochastic path is opt-in through ``ltx25_ancestral_stage1`` and only
    active for stage 1. Stage 2 and non-distilled/base configurations retain the
    deterministic LTX-2 Euler update.
    """

    ANCESTRAL_NOISE_SEED_OFFSET = 10_000

    def __init__(self, config):
        super().__init__(config)
        self._stage = 1
        ancestral_requested = bool(config.get("ltx25_ancestral_stage1", False))
        if ancestral_requested and config.get("pipeline_type") != "distilled":
            raise ValueError("ltx25_ancestral_stage1 is only valid with pipeline_type='distilled'")
        self._use_ancestral_stage1 = ancestral_requested
        self._ancestral_eta = float(config.get("ancestral_eta", config.get("ltx25_ancestral_eta", 1.0)))
        self._ancestral_s_noise = float(config.get("ancestral_s_noise", config.get("ltx25_ancestral_s_noise", 1.0)))
        self._ancestral_noise_seed_offset = int(
            config.get(
                "ancestral_noise_seed_offset",
                config.get(
                    "ltx25_ancestral_noise_seed_offset",
                    self.ANCESTRAL_NOISE_SEED_OFFSET,
                ),
            )
        )
        self._ancestral_generator = None

    @property
    def stage(self) -> int:
        return self._stage

    def set_stage(self, stage: int) -> None:
        if stage not in (1, 2):
            raise ValueError(f"LTX-2.5 stage must be 1 or 2, got {stage}")
        self._stage = stage

    @property
    def use_ancestral_step(self) -> bool:
        return self._use_ancestral_stage1 and self._stage == 1

    def prepare(self, seed: int, *args, **kwargs):
        super().prepare(seed, *args, **kwargs)
        self._ancestral_generator = None
        if self.use_ancestral_step:
            self._ancestral_generator = torch.Generator(device=AI_DEVICE).manual_seed(seed + self._ancestral_noise_seed_offset)

    def _prepare_video_latents(self, *args, **kwargs) -> None:
        super()._prepare_video_latents(*args, **kwargs)

        # LTX-2.5 treats the target's first causal latent frame as a standalone
        # pixel-frame token class. The source pipeline marks it even when there
        # are no generated keyframe slots; ordinary image/reference tokens stay
        # unmarked.
        state = self.video_latent_state
        keyframes_mask = torch.zeros_like(state.denoise_mask)
        _, frames, _, _ = self.video_latent_shape_orig
        main_tokens = self._video_main_num_tokens
        if frames <= 0 or main_tokens is None or main_tokens % frames != 0:
            raise ValueError(f"Cannot determine LTX-2.5 first-frame token count from frames={frames}, main_tokens={main_tokens}")
        keyframes_mask[: main_tokens // frames] = 1.0
        state.keyframes_mask = keyframes_mask

    @staticmethod
    def ancestral_euler_step(
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        noise: torch.Tensor | None,
        *,
        eta: float = 1.0,
        s_noise: float = 1.0,
    ) -> torch.Tensor:
        """Apply the rectified-flow ancestral Euler update used by LTX-2.5."""
        sigma = sigma.to(torch.float32)
        sigma_next = sigma_next.to(torch.float32)
        if sigma_next == 0:
            return denoised_sample.to(sample.dtype)
        if eta > 0 and noise is None:
            raise ValueError("LTX-2.5 ancestral Euler requires noise when eta > 0")

        x = sample.to(torch.float32)
        denoised = denoised_sample.to(torch.float32)
        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio
        sigma_down_ratio = sigma_down / sigma
        x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * denoised

        if eta > 0:
            alpha_next = 1.0 - sigma_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0) ** 0.5
            x_next = (alpha_next / alpha_down) * x_next + noise.to(torch.float32) * s_noise * renoise_coeff
        return x_next.to(sample.dtype)

    def _unpatchify_final_latents(self) -> None:
        _, frames_v, height_v, width_v = self.video_latent_shape_orig
        channels_a, _, mel_bins_a = self.audio_latent_shape_orig

        video_latent = self.video_latent_state.latent
        main_tokens = getattr(self, "_video_main_num_tokens", None)
        if main_tokens is not None and video_latent.shape[0] > main_tokens:
            video_latent = video_latent[:main_tokens]
        self.video_latent_state.latent = self.video_patchifier.unpatchify(
            video_latent,
            frames_v,
            height_v,
            width_v,
        )
        self.audio_latent_state.latent = self.audio_patchifier.unpatchify(
            self.audio_latent_state.latent,
            channels=channels_a,
            mel_bins=mel_bins_a,
        )

    def step_post(self):
        if not self.use_ancestral_step:
            return super().step_post()
        if self._ancestral_generator is None:
            raise RuntimeError("LTX-2.5 ancestral generator is not initialized; call prepare() first")

        video_denoised = self.post_process_latent(
            self.v_noise_pred.float(),
            self.video_latent_state.denoise_mask,
            self.video_latent_state.clean_latent,
        )
        audio_denoised = self.post_process_latent(
            self.a_noise_pred.float(),
            self.audio_latent_state.denoise_mask,
            self.audio_latent_state.clean_latent,
        )

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        if sigma_next == 0:
            self.video_latent_state.latent = video_denoised.to(self.video_latent_state.latent.dtype)
            self.audio_latent_state.latent = audio_denoised.to(self.audio_latent_state.latent.dtype)
            self._unpatchify_final_latents()
            return

        if self._ancestral_eta > 0:
            video_noise = torch.randn(
                self.video_latent_state.latent.shape,
                generator=self._ancestral_generator,
                dtype=self.video_latent_state.latent.dtype,
                device=self.video_latent_state.latent.device,
            )
            audio_noise = torch.randn(
                self.audio_latent_state.latent.shape,
                generator=self._ancestral_generator,
                dtype=self.audio_latent_state.latent.dtype,
                device=self.audio_latent_state.latent.device,
            )
        else:
            video_noise = None
            audio_noise = None

        video_dtype = self.video_latent_state.latent.dtype
        audio_dtype = self.audio_latent_state.latent.dtype
        video_next = self.ancestral_euler_step(
            self.video_latent_state.latent.float(),
            video_denoised,
            sigma,
            sigma_next,
            video_noise,
            eta=self._ancestral_eta,
            s_noise=self._ancestral_s_noise,
        )
        audio_next = self.ancestral_euler_step(
            self.audio_latent_state.latent.float(),
            audio_denoised,
            sigma,
            sigma_next,
            audio_noise,
            eta=self._ancestral_eta,
            s_noise=self._ancestral_s_noise,
        )

        if self._ancestral_eta > 0:
            # Conditioning is re-applied after stochastic noise injection,
            # matching the source ancestral loop.
            video_next = self.post_process_latent(
                video_next,
                self.video_latent_state.denoise_mask,
                self.video_latent_state.clean_latent,
            )
            audio_next = self.post_process_latent(
                audio_next,
                self.audio_latent_state.denoise_mask,
                self.audio_latent_state.clean_latent,
            )
        self.video_latent_state.latent = video_next.to(video_dtype)
        self.audio_latent_state.latent = audio_next.to(audio_dtype)

    def clear(self):
        super().clear()
        self._ancestral_generator = None
