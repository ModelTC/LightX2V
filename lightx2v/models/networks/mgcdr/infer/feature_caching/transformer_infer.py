import numpy as np
from ..transformer_infer import MagicDriveTransformerInfer
import torch
import gc


class MagicDriveTransformerInferTeaCaching(MagicDriveTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    def infer(self, weights, x, y, c, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map):
    
        modulated_inp = t_mlp

        if self.scheduler.cnt % 2 == 0:  # even -> conditon
            self.scheduler.is_even = True
            if self.scheduler.cnt < self.scheduler.ret_steps or self.scheduler.cnt >= self.scheduler.cutoff_steps:
                should_calc_even = True
                self.scheduler.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.scheduler.coefficients)
                self.scheduler.accumulated_rel_l1_distance_even += rescale_func(
                    ((modulated_inp - self.scheduler.previous_e0_even.cuda()).abs().mean() / self.scheduler.previous_e0_even.cuda().abs().mean()).cpu().item()
                )
                if self.scheduler.accumulated_rel_l1_distance_even < self.scheduler.teacache_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.scheduler.accumulated_rel_l1_distance_even = 0
            self.scheduler.previous_e0_even = modulated_inp.clone()
            if self.config["cpu_offload"]:
                self.scheduler.previous_e0_even = self.scheduler.previous_e0_even.cpu()
                modulated_inp = modulated_inp.cpu()
                del modulated_inp
                torch.cuda.empty_cache()
                gc.collect()

        else:  # odd -> unconditon
            self.scheduler.is_even = False
            if self.scheduler.cnt < self.scheduler.ret_steps or self.scheduler.cnt >= self.scheduler.cutoff_steps:
                should_calc_odd = True
                self.scheduler.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.scheduler.coefficients)
                self.scheduler.accumulated_rel_l1_distance_odd += rescale_func(
                    ((modulated_inp - self.scheduler.previous_e0_odd.cuda()).abs().mean() / self.scheduler.previous_e0_odd.cuda().abs().mean()).cpu().item()
                )
                if self.scheduler.accumulated_rel_l1_distance_odd < self.scheduler.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.scheduler.accumulated_rel_l1_distance_odd = 0
            self.scheduler.previous_e0_odd = modulated_inp.clone()
            if self.config["cpu_offload"]:
                self.scheduler.previous_e0_odd = self.scheduler.previous_e0_odd.cpu()
                modulated_inp = modulated_inp.cpu()
                del modulated_inp
                torch.cuda.empty_cache()
                gc.collect()

        if self.scheduler.is_even:
            if not should_calc_even:
                x += self.scheduler.previous_residual_even.cuda()
            else:
                ori_x = x.clone()
                x = super().infer(weights, x, y, c, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map, is_cfg)
                self.scheduler.previous_residual_even = x - ori_x
                if self.config["cpu_offload"]:
                    self.scheduler.previous_residual_even = self.scheduler.previous_residual_even.cpu()
                    ori_x = ori_x.to("cpu")
                    del ori_x
                    torch.cuda.empty_cache()
                    gc.collect()
        else:
            if not should_calc_odd:
                x += self.scheduler.previous_residual_odd.cuda()
            else:
                ori_x = x.clone()
                x = super().infer(weights, x, y, c, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map, is_cfg)
                self.scheduler.previous_residual_odd = x - ori_x
                if self.config["cpu_offload"]:
                    self.scheduler.previous_residual_odd = self.scheduler.previous_residual_odd.cpu()
                    ori_x = ori_x.to("cpu")
                    del ori_x
                    torch.cuda.empty_cache()
                    gc.collect()

        return x
