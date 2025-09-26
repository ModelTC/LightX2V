from lightx2v.utils.envs import *


class BaseScheduler:
    def __init__(self, config):
        self.config = config
        self.latents = None
        self.step_index = 0
        self.infer_steps = config["infer_steps"]
        self.caching_records = [True] * config["infer_steps"]
        self.flag_df = False
        self.transformer_infer = None
        self.infer_condition = True  # cfg status

    def step_pre(self, step_index):
        self.step_index = step_index
        if GET_DTYPE() == GET_SENSITIVE_DTYPE():
            self.latents = self.latents.to(GET_DTYPE())

    def clear(self):
        pass
