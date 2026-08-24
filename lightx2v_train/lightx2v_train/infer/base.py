import os

import torch

from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler


class BaseInferencer:
    def __init__(self, config):
        self.config = config
        self.infer_config = config.get("inference", {})
        self.output_infer_dir = self.infer_config.get("output_dir", None)
        if self.output_infer_dir is not None:
            os.makedirs(self.output_infer_dir, exist_ok=True)

        self.model = None
        self.dataloader_eval = None
        self.enable_cfg = True
        self.guidance_scale = None

        self.scheduler = RectifiedFlowMatchingScheduler(config)

    def set_data(self, dataloader_val):
        self.dataloader_eval = dataloader_val

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
