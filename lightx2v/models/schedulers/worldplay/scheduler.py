from lightx2v.models.schedulers.worldplay.ar_scheduler import WorldPlayARScheduler


class WorldPlayDistillScheduler(WorldPlayARScheduler):
    """AR flow matching with the configured step count and shift."""

    def step_post(self):
        latent_dtype = self.latents.dtype
        super().step_post()
        self.latents = self.latents.to(latent_dtype)
