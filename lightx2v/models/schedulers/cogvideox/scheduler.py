import numpy as np
import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from lightx2v.models.schedulers.scheduler import BaseScheduler


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def rescale_zero_terminal_snr(alphas_cumprod):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """

    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt

    return alphas_bar


class CogvideoxXDPMScheduler(BaseScheduler):
    def __init__(self, config):
        self.config = config
        self.set_timesteps()
        self.generator = torch.Generator().manual_seed(config.seed)
        self.noise_pred = None

        if self.config.beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(self.config.scheduler_beta_start**0.5, self.config.scheduler_beta_end**0.5, self.config.num_train_timesteps, dtype=torch.float64) ** 2
        else:
            raise NotImplementedError(f"{self.config.beta_schedule} is not implemented for {self.__class__}")
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(torch.device("cuda"))

        # Modify: SNR shift following SD3
        self.alphas_cumprod = self.alphas_cumprod / (self.config.scheduler_snr_shift_scale + (1 - self.config.scheduler_snr_shift_scale) * self.alphas_cumprod)

        # Rescale for zero SNR
        if self.config.scheduler_rescale_betas_zero_snr:
            self.alphas_cumprod = rescale_zero_terminal_snr(self.alphas_cumprod)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if self.config.scheduler_set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

    def scale_model_input(self, sample, timestep=None):
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(self):
        if self.config.num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {self.config.num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )
        self.infer_steps = self.config.num_inference_steps
        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, self.config.num_inference_steps).round()[::-1].copy().astype(np.int64)
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.config.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, self.infer_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.config.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'.")
        self.timesteps = torch.Tensor(timesteps).to(torch.device("cuda")).int()

    def prepare(self, image_encoder_output):
        self.image_encoder_output = image_encoder_output
        self.prepare_latents(shape=self.config.target_shape, dtype=torch.bfloat16)
        self.prepare_guidance()
        self.prepare_rotary_pos_embedding()

    def prepare_latents(self, shape, dtype):
        latents = randn_tensor(shape, generator=self.generator, device=torch.device("cuda"), dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.init_noise_sigma
        self.latents = latents
        self.old_pred_original_sample = None

    def prepare_guidance(self):
        self.guidance_scale = self.config.guidance_scale

    def prepare_rotary_pos_embedding(self):
        grid_height = self.config.height // (self.config.vae_scale_factor_spatial * self.config.patch_size)
        grid_width = self.config.width // (self.config.vae_scale_factor_spatial * self.config.patch_size)

        p = self.config.patch_size
        p_t = self.config.patch_size_t

        base_size_width = self.config.transformer_sample_width // p
        base_size_height = self.config.transformer_sample_height // p

        num_frames = self.latents.size(1)

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=torch.device("cuda"),
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.config.transformer_attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=torch.device("cuda"),
            )

        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin
        self.image_rotary_emb = (freqs_cos, freqs_sin) if self.config.use_rotary_positional_embeddings else None

    def get_variables(self, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back=None):
        lamb = ((alpha_prod_t / (1 - alpha_prod_t)) ** 0.5).log()
        lamb_next = ((alpha_prod_t_prev / (1 - alpha_prod_t_prev)) ** 0.5).log()
        h = lamb_next - lamb

        if alpha_prod_t_back is not None:
            lamb_previous = ((alpha_prod_t_back / (1 - alpha_prod_t_back)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back):
        mult1 = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * (-h).exp()
        mult2 = (-2 * h).expm1() * alpha_prod_t_prev**0.5

        if alpha_prod_t_back is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def step_post(self):
        if self.infer_steps is None:
            raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        timestep = self.timesteps[self.step_index]
        timestep_back = self.timesteps[self.step_index - 1] if self.step_index > 0 else None
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.infer_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t_back = self.alphas_cumprod[timestep_back] if timestep_back is not None else None

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # To make style tests pass, commented out `pred_epsilon` as it is an unused variable
        if self.config.scheduler_prediction_type == "epsilon":
            pred_original_sample = (self.latents - beta_prod_t ** (0.5) * self.noise_pred) / alpha_prod_t ** (0.5)
            # pred_epsilon = model_output
        elif self.config.scheduler_prediction_type == "sample":
            pred_original_sample = self.noise_pred
            # pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.scheduler_prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * self.latents - (beta_prod_t**0.5) * self.noise_pred
            # pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(f"prediction_type given as {self.config.scheduler_prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`")

        h, r, lamb, lamb_next = self.get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
        mult = list(self.get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back))
        mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

        noise = randn_tensor(self.latents.shape, generator=self.generator, device=self.latents.device, dtype=self.latents.dtype)
        prev_sample = mult[0] * self.latents - mult[1] * pred_original_sample + mult_noise * noise

        if self.old_pred_original_sample is None or prev_timestep < 0:
            # Save a network evaluation if all noise levels are 0 or on the first step
            self.latents = prev_sample
            self.old_pred_original_sample = pred_original_sample
        else:
            denoised_d = mult[2] * pred_original_sample - mult[3] * self.old_pred_original_sample
            noise = randn_tensor(self.latents.shape, generator=self.generator, device=self.latents.device, dtype=self.latents.dtype)
            x_advanced = mult[0] * self.latents - mult[1] * denoised_d + mult_noise * noise
            self.latents = x_advanced
            self.old_pred_original_sample = pred_original_sample

        self.latents = self.latents.to(torch.bfloat16)
