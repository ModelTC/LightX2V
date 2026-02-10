from lightx2v.utils.envs import *


class BaseScheduler:
    def __init__(self, config):
        self.config = config
        self.latents = None
        self.step_index = 0
        self.flag_df = False
        self.transformer_infer = None
        self.infer_condition = True  # cfg status
        self.keep_latents_dtype_in_scheduler = False

    def step_pre(self, step_index):
        self.step_index = step_index
        if GET_DTYPE() == GET_SENSITIVE_DTYPE() and not self.keep_latents_dtype_in_scheduler:
            self.latents = self.latents.to(GET_DTYPE())

    def clear(self):
        pass
