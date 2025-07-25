import torch
from ..scheduler import MagicDriverScheduler


class MagicDriverSchedulerTeaCaching(MagicDriverScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.cnt = 0
        self.num_steps = self.config.infer_steps * 2
        self.teacache_thresh = self.config.teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.use_ret_steps = self.config.use_ret_steps
        self.coefficients = self.config.coefficients[0]
        self.ret_steps = 1 * 2
        self.cutoff_steps = self.config.infer_steps * 2 - 2

    def clear(self):
        if self.previous_e0_even is not None:
            self.previous_e0_even = self.previous_e0_even.cpu()
        if self.previous_e0_odd is not None:
            self.previous_e0_odd = self.previous_e0_odd.cpu()
        if self.previous_residual_even is not None:
            self.previous_residual_even = self.previous_residual_even.cpu()
        if self.previous_residual_odd is not None:
            self.previous_residual_odd = self.previous_residual_odd.cpu()
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        torch.cuda.empty_cache()