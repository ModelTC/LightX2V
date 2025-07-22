import json
import torch
from typing import List, Optional, Tuple, Union
from lightx2v.models.schedulers.scheduler import BaseScheduler


class MagicDriverScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device("cuda")
        self.infer_steps = self.config.get("infer_steps", 30)
        # self.target_video_length = self.config.get("target_video_length")
        self.num_timesteps = self.config.get("num_timesteps", 1000)
        self.noise_added = None
    
    def prepare_latents(self, target_shape, dtype=torch.bfloat16):
        self.latents = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=self.device,
            generator=self.generator
        )
        
    def prepare(self):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)
        self.prepare_latents(self.config.target_shape, dtype=torch.bfloat16)
        # self.set_timesteps(self.infer_steps, device=self.device)
        
    # def set_timesteps(
    #     self,
    #     infer_steps: Union[int, None] = None,
    #     device: Union[str, torch.device] = None,
    # ):
    #     timesteps = [(1.0 - i / infer_steps) * self.num_timesteps for i in range(infer_steps)]
    #     timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
    #     self.timesteps = timesteps

    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])
        return timepoints * original_samples + (1 - timepoints) * noise
    
    # def step_pre(self, step_index, mask):
    #     super().step_pre(step_index)
    #     if self.noise_added is None:
    #         self.noise_added = torch.zeros_like(mask, dtype=torch.bool)
    #         self.noise_added = self.noise_added | (mask == 1)
            
    #     t = self.timesteps[step_index]
    #     mask_t = mask * self.num_timesteps
    #     self.ori_latents = self.latents.clone()
    #     latents_noise = self.add_noise(self.ori_latents, torch.randn_like(self.ori_latents), t)
    #     mask_t_upper = mask_t >= t.unsqueeze(1)
    #     latents_mask = mask_t_upper.repeat(2, 1)
    #     mask_add_noise = mask_t_upper & ~self.noise_added # all False
    #     self.latents = torch.where(mask_add_noise[:, None, :, None, None], latents_noise, self.ori_latents)
    #     self.noise_added = mask_t_upper
    
    def step_pre(self, step_index):
        self.step_index = step_index
        
    def step_post(self, ):
        pass
  