import os

import torch

from lightx2v_train.data.utils import preserve_cache_dtype
from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler


def move_cached_value(value, model, key=None):
    if torch.is_tensor(value):
        kwargs = {"device": model.device}
        if value.is_floating_point() and not preserve_cache_dtype(key):
            kwargs["dtype"] = model.running_dtype
        return value.to(**kwargs)
    if isinstance(value, dict):
        return {name: move_cached_value(item, model, name) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_cached_value(item, model, key) for item in value)
    if isinstance(value, list):
        return [move_cached_value(item, model, key) for item in value]
    return value


def cached_condition(sample, model, role):
    conditioning = sample.get("conditioning", {})
    cached = conditioning.get(role)
    if cached is None:
        return None
    condition = dict(conditioning.get("shared", {}))
    condition.update(cached)
    return move_cached_value(condition, model)


class BaseInferencer:
    def __init__(self, config):
        self.config = config
        self.infer_config = config.get("inference", {})
        self.output_infer_dir = self.infer_config.get("output_dir", None)
        if self.output_infer_dir is not None:
            os.makedirs(self.output_infer_dir, exist_ok=True)

        self.model = None
        self.dataloader_val = None
        self.enable_cfg = True
        self.guidance_scale = None

        self.scheduler = RectifiedFlowMatchingScheduler(config)

    def set_data(self, dataloader_val):
        self.dataloader_val = dataloader_val

    def set_model(self, model):
        self.model = model

    def cfg_guided_denoise(
        self,
        latents,
        timestep_or_sigma,
        pos_cond,
        neg_cond,
        model=None,
    ):
        model = self.model if model is None else model
        denoiser_input = model.prepare_denoiser_input(
            latents,
            condition=pos_cond,
        )

        pred_pos = model.denoise(
            denoiser_input,
            timestep_or_sigma,
            pos_cond,
        )

        if self.enable_cfg:
            pred_neg = model.denoise(
                denoiser_input,
                timestep_or_sigma,
                neg_cond,
            )
            pred = model.apply_cfg(pred_pos, pred_neg, self.guidance_scale)
        else:
            pred = pred_pos
        return model.postprocess_denoiser_output(pred, denoiser_input)

    @torch.no_grad()
    def infer(self):
        raise NotImplementedError
