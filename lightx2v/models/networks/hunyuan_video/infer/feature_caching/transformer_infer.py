import numpy as np
import torch

from lightx2v.models.networks.hunyuan_video.infer.offload.transformer_infer import HunyuanVideo15OffloadTransformerInfer


class HunyuanTransformerInferTeaCaching(HunyuanVideo15OffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.teacache_thresh = self.config["teacache_thresh"]
        self.coefficients = self.config["coefficients"]

        self.accumulated_rel_l1_distance_odd = 0
        self.previous_modulated_input_odd = None
        self.previous_residual_odd = None

        self.accumulated_rel_l1_distance_even = 0
        self.previous_modulated_input_even = None
        self.previous_residual_even = None

    def calculate_should_calc(self, img, vec, block):
        inp = img.clone()
        vec_ = vec.clone()
        img_mod_layer = block.img_branch.img_mod
        if self.config["cpu_offload"]:
            img_mod_layer.to_cuda()

        img_mod1_shift, img_mod1_scale, _, _, _, _ = img_mod_layer.apply(vec_).chunk(6, dim=-1)
        inp = inp.squeeze(0)
        normed_inp = torch.nn.functional.layer_norm(inp, (inp.shape[1],), None, None, 1e-6)
        modulated_inp = normed_inp * (1 + img_mod1_scale) + img_mod1_shift

        del normed_inp, inp, vec_

        if self.scheduler.step_index == 0 or self.scheduler.step_index == self.scheduler.infer_steps - 1:
            should_calc = True
            if self.scheduler.infer_condition:
                self.accumulated_rel_l1_distance_odd = 0
                self.previous_modulated_input_odd = modulated_inp
            else:
                self.accumulated_rel_l1_distance_even = 0
                self.previous_modulated_input_even = modulated_inp
        else:
            rescale_func = np.poly1d(self.coefficients)
            if self.scheduler.infer_condition:
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp - self.previous_modulated_input_odd).abs().mean() / self.previous_modulated_input_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_odd = 0
                self.previous_modulated_input_odd = modulated_inp
            else:
                self.accumulated_rel_l1_distance_even += rescale_func(
                    ((modulated_inp - self.previous_modulated_input_even).abs().mean() / self.previous_modulated_input_even.abs().mean()).cpu().item()
                )
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_even = 0

                self.previous_modulated_input_even = modulated_inp
            del modulated_inp

        return should_calc

    def infer(self, weights, infer_module_out):
        should_calc = self.calculate_should_calc(infer_module_out.img, infer_module_out.vec, weights.double_blocks[0])
        if not should_calc:
            if self.scheduler.infer_condition:
                infer_module_out.img += self.previous_residual_odd
            else:
                infer_module_out.img += self.previous_residual_even
        else:
            ori_img = infer_module_out.img.clone()

            self.infer_func(weights, infer_module_out)

            if self.scheduler.infer_condition:
                self.previous_residual_odd = infer_module_out.img - ori_img
            else:
                self.previous_residual_even = infer_module_out.img - ori_img

        x = self.infer_final_layer(weights, infer_module_out)
        return x

    def clear(self):
        if self.previous_residua_odd is not None:
            self.previous_residual_odd = self.previous_residual_odd.cpu()

        if self.previous_modulated_input_odd is not None:
            self.previous_modulated_input_odd = self.previous_modulated_input_odd.cpu()

        if self.previous_residual_even is not None:
            self.previous_residual_even = self.previous_residual_even.cpu()

        if self.previous_modulated_input_even is not None:
            self.previous_modulated_input_even = self.previous_modulated_input_even.cpu()

        self.previous_modulated_input_odd = None
        self.previous_residual_odd = None
        self.previous_modulated_input_even = None
        self.previous_residual_even = None
        torch.cuda.empty_cache()
