import gc
import json

import numpy as np
import torch
import torch.nn.functional as F

from lightx2v.models.networks.hunyuan_video.infer.offload.transformer_infer import HunyuanVideo15OffloadTransformerInfer


class HunyuanVideo15TransformerInferMagCaching(HunyuanVideo15OffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.magcache_thresh = config.get("magcache_thresh", 0.2)
        self.K = config.get("magcache_K", 6)
        self.retention_ratio = config.get("magcache_retention_ratio", 0.2)
        self.mag_ratios = np.array(config.get("magcache_ratios", []))
        self.enable_magcache_calibration = config.get("magcache_calibration", True)
        # {True: cond_param, False: uncond_param}
        self.accumulated_err = {True: 0.0, False: 0.0}
        self.accumulated_steps = {True: 0, False: 0}
        self.accumulated_ratio = {True: 1.0, False: 1.0}
        self.residual_cache = {True: None, False: None}
        self.residual_cache_txt = {True: None, False: None}
        # calibration args
        self.norm_ratio = [[1.0], [1.0]]  # mean of magnitude ratio
        self.norm_std = [[0.0], [0.0]]  # std of magnitude ratio
        self.cos_dis = [[0.0], [0.0]]  # cosine distance of residual features

    @torch.no_grad()
    def infer(self, weights, infer_module_out):
        skip_forward = False
        step_index = self.scheduler.step_index
        infer_condition = self.scheduler.infer_condition

        if self.enable_magcache_calibration:
            skip_forward = False
        else:
            if step_index >= int(self.config["infer_steps"] * self.retention_ratio):
                # conditional and unconditional in one list
                cur_mag_ratio = self.mag_ratios[0][step_index] if infer_condition else self.mag_ratios[1][step_index]
                # magnitude ratio between current step and the cached step
                self.accumulated_ratio[infer_condition] = self.accumulated_ratio[infer_condition] * cur_mag_ratio
                self.accumulated_steps[infer_condition] += 1  # skip steps plus 1
                # skip error of current steps
                cur_skip_err = np.abs(1 - self.accumulated_ratio[infer_condition])
                # accumulated error of multiple steps
                self.accumulated_err[infer_condition] += cur_skip_err

                if self.accumulated_err[infer_condition] < self.magcache_thresh and self.accumulated_steps[infer_condition] <= self.K:
                    skip_forward = True
                else:
                    self.accumulated_err[infer_condition] = 0
                    self.accumulated_steps[infer_condition] = 0
                    self.accumulated_ratio[infer_condition] = 1.0

        if not skip_forward:
            self.infer_calculating(weights, infer_module_out)
        else:
            self.infer_using_cache(infer_module_out)

        x = self.infer_final_layer(weights, infer_module_out)

        return x

    def infer_calculating(self, weights, infer_module_out):
        step_index = self.scheduler.step_index
        infer_condition = self.scheduler.infer_condition

        ori_img = infer_module_out.img.clone()
        ori_txt = infer_module_out.txt.clone()
        self.infer_func(weights, infer_module_out)

        previous_residual = infer_module_out.img - ori_img
        previous_residual_txt = infer_module_out.txt - ori_txt

        if self.config["cpu_offload"]:
            previous_residual = previous_residual.cpu()

        if self.enable_magcache_calibration and step_index >= 1:
            norm_ratio = ((previous_residual.norm(dim=-1) / self.residual_cache[infer_condition].norm(dim=-1)).mean()).item()
            norm_std = (previous_residual.norm(dim=-1) / self.residual_cache[infer_condition].norm(dim=-1)).std().item()
            cos_dis = (1 - F.cosine_similarity(previous_residual, self.residual_cache[infer_condition], dim=-1, eps=1e-8)).mean().item()
            _index = int(not infer_condition)
            self.norm_ratio[_index].append(round(norm_ratio, 5))
            self.norm_std[_index].append(round(norm_std, 5))
            self.cos_dis[_index].append(round(cos_dis, 5))
            print(f"time: {step_index}, infer_condition: {infer_condition}, norm_ratio: {norm_ratio}, norm_std: {norm_std}, cos_dis: {cos_dis}")

        self.residual_cache[infer_condition] = previous_residual
        self.residual_cache_txt[infer_condition] = previous_residual_txt

        if self.config["cpu_offload"]:
            ori_img = ori_img.to("cpu")
            ori_txt = ori_txt.to("cpu")
            del ori_img, ori_txt
            torch.cuda.empty_cache()
            gc.collect()

    def infer_using_cache(self, infer_module_out):
        residual_img = self.residual_cache[self.scheduler.infer_condition]
        residual_txt = self.residual_cache_txt[self.scheduler.infer_condition]
        infer_module_out.img.add_(residual_img.cuda())
        infer_module_out.txt.add_(residual_txt.cuda())

    def clear(self):
        self.accumulated_err = {True: 0.0, False: 0.0}
        self.accumulated_steps = {True: 0, False: 0}
        self.accumulated_ratio = {True: 1.0, False: 1.0}
        self.residual_cache = {True: None, False: None}
        self.residual_cache_txt = {True: None, False: None}
        if self.enable_magcache_calibration:
            print("norm ratio")
            print(self.norm_ratio)
            print("norm std")
            print(self.norm_std)
            print("cos_dis")
            print(self.cos_dis)

            def save_json(filename, obj_list):
                with open(filename + ".json", "w") as f:
                    json.dump(obj_list, f)

            save_json("mag_ratio", self.norm_ratio)
            save_json("mag_std", self.norm_std)
            save_json("cos_dis", self.cos_dis)
        torch.cuda.empty_cache()
