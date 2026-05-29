import torch


class ErnieImageTransformerModel:
    def __init__(self, config, transformer, diffusers_cpu_offload=False):
        self.config = config
        self.transformer = transformer
        self.diffusers_cpu_offload = diffusers_cpu_offload
        self.scheduler = None

    @property
    def dtype(self):
        return getattr(self.transformer, "dtype", None)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @staticmethod
    def _timestep_batch(timestep, batch_size, device, dtype):
        timestep_value = timestep.item() if hasattr(timestep, "item") else timestep
        return torch.full((batch_size,), timestep_value, device=device, dtype=dtype)

    def infer(self, inputs):
        if self.scheduler is None:
            raise ValueError("ERNIE-Image transformer requires a scheduler before inference.")
        if self.scheduler.latents is None:
            raise ValueError("ERNIE-Image scheduler latents are not prepared.")

        latents = self.scheduler.latents
        do_cfg = inputs["do_classifier_free_guidance"]
        total_batch_size = int(inputs["total_batch_size"])

        if do_cfg:
            latent_model_input = torch.cat([latents, latents], dim=0)
            timestep_batch_size = total_batch_size * 2
        else:
            latent_model_input = latents
            timestep_batch_size = total_batch_size

        timestep_batch = self._timestep_batch(
            inputs["timestep"],
            timestep_batch_size,
            inputs["device"],
            inputs["dtype"],
        )
        pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep_batch,
            text_bth=inputs["text_bth"],
            text_lens=inputs["text_lens"],
            return_dict=False,
        )[0]

        if do_cfg:
            pred_uncond, pred_cond = pred.chunk(2, dim=0)
            pred = pred_uncond + float(inputs["guidance_scale"]) * (pred_cond - pred_uncond)

        self.scheduler.noise_pred = pred
        return pred
