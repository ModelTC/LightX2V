import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class LingbotVAFlowMatchScheduler(BaseScheduler):
    """Flow-match Euler scheduler used by LingBot-VA video/action loops."""

    def __init__(self, config, *, shift_key="sample_shift", infer_steps_key="infer_steps", extra_one_step=True):
        super().__init__({"infer_steps": int(config[infer_steps_key])})
        self.num_train_timesteps = 1000
        self.sample_shift = float(config[shift_key])
        self.sigma_max = 1.0
        self.sigma_min = 0.0
        self.extra_one_step = bool(extra_one_step)
        self.loop_inputs = None
        self.step_input_builder = None
        self.noise_pred_processor = None
        self.set_timesteps(self.infer_steps)

    def set_timesteps(self, infer_steps=None, denoising_strength=1.0, device=None):
        infer_steps = self.infer_steps if infer_steps is None else int(infer_steps)
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            sigmas = torch.linspace(sigma_start, self.sigma_min, infer_steps + 1)[:-1]
        else:
            sigmas = torch.linspace(sigma_start, self.sigma_min, infer_steps)
        self.sigmas = self.sample_shift * sigmas / (1 + (self.sample_shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if device is not None:
            self.timesteps = self.timesteps.to(device)
        self.infer_steps = int(self.timesteps.numel())

    def bind_step_inputs(self, inputs, input_builder):
        self.loop_inputs = inputs
        self.step_input_builder = input_builder

    def bind_noise_pred_processor(self, noise_pred_processor):
        self.noise_pred_processor = noise_pred_processor

    def prepare_loop(
        self,
        infer_steps,
        device,
        latent_shape=None,
        latents=None,
        seed=None,
        dtype=None,
        cond_latent=None,
        step_latents=True,
        video_exec_step=-1,
    ):
        if latents is None:
            if latent_shape is None:
                raise ValueError("LingbotVAFlowMatchScheduler.prepare_loop requires latents or latent_shape.")
            self.prepare_latents(seed=seed, latent_shape=latent_shape, device=device, dtype=dtype)
        else:
            self.latents = latents
        self.cond_latent = cond_latent
        self.step_latents = step_latents
        self.set_timesteps(infer_steps, device=device)
        self.loop_timesteps = self.padded_timesteps(device=device)
        if video_exec_step != -1:
            self.loop_timesteps = self.loop_timesteps[:video_exec_step]
        self.infer_steps = len(self.loop_timesteps)
        self.noise_pred = None

    def prepare_latents(self, seed, latent_shape, device, dtype):
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.latents = torch.randn(
            *latent_shape,
            generator=self.generator,
            device=device,
            dtype=dtype,
        )

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.current_timestep = self.loop_timesteps[step_index]
        self.last_step = step_index == len(self.loop_timesteps) - 1
        if self.loop_inputs is not None and self.step_input_builder is not None:
            self.loop_inputs.clear()
            self.loop_inputs.update(self.step_input_builder(self))

    def step_post(self):
        if self.noise_pred is None:
            raise RuntimeError("LingbotVAFlowMatchScheduler requires noise_pred before step_post().")
        if self.step_latents:
            if self.noise_pred_processor is not None:
                self.noise_pred = self.noise_pred_processor(self.noise_pred)
            self.latents = self.step(self.noise_pred, self.current_timestep, self.latents, return_dict=False)
        if self.cond_latent is not None:
            self.latents[:, :, 0:1] = self.cond_latent
        self.noise_pred = None

    def step(self, model_output, timestep, sample, return_dict=False):
        step_index = getattr(self, "step_index", None)
        if step_index is not None:
            timestep_id = step_index
        else:
            if isinstance(timestep, torch.Tensor):
                timestep_ref = timestep.detach().to(device=self.timesteps.device, dtype=self.timesteps.dtype)
            else:
                timestep_ref = torch.tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)
            timestep_id = int(torch.argmin((self.timesteps - timestep_ref).abs()).item())

        sigma = self.sigmas[timestep_id].to(sample.device, sample.dtype)
        if timestep_id + 1 >= len(self.timesteps):
            sigma_next = torch.zeros((), device=sample.device, dtype=sample.dtype)
        else:
            sigma_next = self.sigmas[timestep_id + 1].to(sample.device, sample.dtype)
        sample = sample + model_output * (sigma_next - sigma)
        if return_dict:
            return {"prev_sample": sample}
        return sample

    def padded_timesteps(self, device=AI_DEVICE, pad_value=0):
        return torch.nn.functional.pad(self.timesteps.to(device), (0, 1), mode="constant", value=pad_value)

    @staticmethod
    def seq_to_patch(patch_size, data_seq, latent_num_frames, latent_height, latent_width, batch_size=1):
        p_t, p_h, p_w = patch_size
        post_f = latent_num_frames // p_t
        post_h = latent_height // p_h
        post_w = latent_width // p_w
        data_patch = data_seq.reshape(batch_size, post_f, post_h, post_w, p_t, p_h, p_w, -1)
        data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def clear(self):
        self.sigmas = None
        self.timesteps = None
        self.loop_timesteps = None
        self.latents = None
        self.cond_latent = None
        self.noise_pred = None
        self.loop_inputs = None
        self.step_input_builder = None
        self.noise_pred_processor = None
