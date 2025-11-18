import torch
import gc
import json
import numpy as np

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
        # calibration args
        self.norm_ratio = [[1.0], [1.0]]  # mean of magnitude ratio
        self.norm_std = [[0.0], [0.0]]  # std of magnitude ratio
        self.cos_dis = [[0.0], [0.0]]  # cosine distance of residual features


    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
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
            x = self.infer_calculating(weights, pre_infer_out)
        else:
            x = self.infer_using_cache(pre_infer_out.x)

        torch.cuda.empty_cache()

        return x

    def infer_calculating(self, weights, pre_infer_out):
        step_index = self.scheduler.step_index
        infer_condition = self.scheduler.infer_condition

        ori_x = pre_infer_out.x.clone()

        x = self.infer_func(weights, pre_infer_out)

        previous_residual = x - ori_x
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

        if self.config["cpu_offload"]:
            ori_x = ori_x.to("cpu")
            del ori_x
            torch.cuda.empty_cache()
            gc.collect()
        return x

    def infer_using_cache(self, x):
        residual_x = self.residual_cache[self.scheduler.infer_condition]
        x.add_(residual_x.cuda())
        return x

    def clear(self):
        self.accumulated_err = {True: 0.0, False: 0.0}
        self.accumulated_steps = {True: 0, False: 0}
        self.accumulated_ratio = {True: 1.0, False: 1.0}
        self.residual_cache = {True: None, False: None}
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
