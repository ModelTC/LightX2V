import torch
from lightx2v.utils.envs import *


class BaseScheduler:
    def __init__(self, config):
        self.config = config
        self.step_index = 0
        self.latents = None
        self.infer_steps = config.infer_steps
        self.caching_records = [True] * config.infer_steps
        self.flag_df = False
        self.transformer_infer = None

    def step_pre(self, step_index):
        self.step_index = step_index
        if GET_DTYPE() == "BF16":
            self.latents = self.latents.to(dtype=torch.bfloat16)

    def clear(self):
        pass
