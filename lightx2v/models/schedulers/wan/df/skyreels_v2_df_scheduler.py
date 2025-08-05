import math
import os

import numpy as np
import torch

from lightx2v.models.schedulers.wan.scheduler import WanScheduler


class WanSkyreelsV2DFScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.df_schedulers = []
        self.flag_df = True

    def prepare(self, image_encoder_output=None):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)

        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if os.path.isfile(self.config.image_path):
            self.seq_len = ((self.config.target_video_length - 1) // self.config.vae_stride[0] + 1) * self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2])
        else:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])

        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift)

    def generate_timestep_matrix(
        self,
        num_frames,
        base_num_frames,
        addnoise_condition,
        num_pre_ready,
        casual_block_size=1,
        ar_step=0,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        self.addnoise_condition = addnoise_condition
        self.predix_video_latent_length = num_pre_ready

        step_template = self.timesteps

        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert ar_step >= min_ar_step, f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, casual_block_size, num_frames_block, base_num_frames_block)
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        while not torch.all(pre_row >= (num_iterations - 1)):
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (num_iterations - 1):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append((new_row != pre_row) & (new_row != num_iterations))  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            valid_interval = [(s * casual_block_size, e * casual_block_size) for s, e in valid_interval]

        self.step_matrix = step_matrix
        self.step_update_mask = step_update_mask
        self.valid_interval = valid_interval
        self.df_timesteps = torch.zeros_like(self.step_matrix)

        self.df_schedulers = []

        for _ in range(base_num_frames):
            sample_scheduler = WanScheduler(self.config)
            sample_scheduler.prepare()
            self.df_schedulers.append(sample_scheduler)

    def step_pre(self, step_index):
        self.step_index = step_index
        self.latents = self.latents.to(dtype=torch.bfloat16)

        valid_interval_start, valid_interval_end = self.valid_interval[step_index]
        timestep = self.step_matrix[step_index][None, valid_interval_start:valid_interval_end].clone()

        if self.addnoise_condition > 0 and valid_interval_start < self.predix_video_latent_length:
            latent_model_input = self.latents[:, valid_interval_start:valid_interval_end, :, :].clone()

            noise_factor = 0.001 * self.addnoise_condition

            self.latents[:, valid_interval_start : self.predix_video_latent_length] = (
                latent_model_input[:, valid_interval_start : self.predix_video_latent_length] * (1.0 - noise_factor)
                + torch.randn_like(latent_model_input[:, valid_interval_start : self.predix_video_latent_length]) * noise_factor
            )
            timestep[:, valid_interval_start : self.predix_video_latent_length] = self.addnoise_condition

        self.df_timesteps[step_index] = timestep

    def step_post(self):
        update_mask_i = self.step_update_mask[self.step_index]
        valid_interval_start, valid_interval_end = self.valid_interval[self.step_index]

        timestep = self.df_timesteps[self.step_index]

        for idx in range(valid_interval_start, valid_interval_end):  # 每一帧单独step
            if update_mask_i[idx].item():
                self.df_schedulers[idx].step_pre(step_index=self.step_index)
                self.df_schedulers[idx].noise_pred = self.noise_pred[:, idx - valid_interval_start]
                self.df_schedulers[idx].timesteps[self.step_index] = timestep[idx]
                self.df_schedulers[idx].latents = self.latents[:, idx]
                self.df_schedulers[idx].step_post()

                self.latents[:, idx] = self.df_schedulers[idx].latents
