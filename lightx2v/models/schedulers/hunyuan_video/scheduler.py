import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler


class HunyuanVideo15Scheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device("cuda")
        self.reverse = True
        self.num_train_timesteps = 1000
        self.sample_shift = self.config["sample_shift"]
        self.reorg_token = False
        self.keep_latents_dtype_in_scheduler = True
        self.sample_guide_scale = self.config["sample_guide_scale"]

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.bfloat16)
        self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift)
        self.multitask_mask = self.get_task_mask(self.config["task"], latent_shape[-3])
        self.cond_latents_concat, self.mask_concat = self._prepare_cond_latents_and_mask(self.config["task"], image_encoder_output["cond_latents"], self.latents, self.multitask_mask, self.reorg_token)

    def prepare_latents(self, seed, latent_shape, dtype=torch.bfloat16):
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.latents = torch.randn(
            1,
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=dtype,
            device=self.device,
            generator=self.generator,
        )

    def set_timesteps(self, num_inference_steps, device, shift):
        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        # Apply timestep shift
        if shift != 1.0:
            sigmas = self.sd3_time_shift(sigmas, shift)

        if not self.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(dtype=torch.float32, device=device)

    def sd3_time_shift(self, t: torch.Tensor, shift):
        return (shift * t) / (1 + (shift - 1) * t)

    def get_task_mask(self, task_type, latent_target_length):
        if task_type == "t2v":
            mask = torch.zeros(latent_target_length)
        elif task_type == "i2v":
            mask = torch.zeros(latent_target_length)
            mask[0] = 1.0
        else:
            raise ValueError(f"{task_type} is not supported !")
        return mask

    def _prepare_cond_latents_and_mask(self, task_type, cond_latents, latents, multitask_mask, reorg_token):
        """
        Prepare multitask mask training logic.

        Args:
            task_type: Type of task ("i2v" or "t2v")
            cond_latents: Conditional latents tensor
            latents: Main latents tensor
            multitask_mask: Multitask mask tensor
            reorg_token: Whether to reorganize tokens

        Returns:
            tuple: (latents_concat, mask_concat) - may contain None values
        """
        latents_concat = None
        mask_concat = None

        if cond_latents is not None and task_type == "i2v":
            latents_concat = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0
        else:
            if reorg_token:
                latents_concat = torch.zeros(latents.shape[0], latents.shape[1] // 2, latents.shape[2], latents.shape[3], latents.shape[4]).to(latents.device)
            else:
                latents_concat = torch.zeros(latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3], latents.shape[4]).to(latents.device)

        mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        mask_concat = self.merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=latents.device)

        return latents_concat, mask_concat

    def merge_tensor_by_mask(self, tensor_1, tensor_2, mask, dim):
        assert tensor_1.shape == tensor_2.shape
        # Mask is a 0/1 vector. Choose tensor_2 when the value is 1; otherwise, tensor_1
        masked_indices = torch.nonzero(mask).squeeze(1)
        tmp = tensor_1.clone()
        if dim == 0:
            tmp[masked_indices] = tensor_2[masked_indices]
        elif dim == 1:
            tmp[:, masked_indices] = tensor_2[:, masked_indices]
        elif dim == 2:
            tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
        return tmp

    def step_post(self):
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        self.latents = sample + model_output * dt
