import gc

import torch

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.ernie_image.infer.post_infer import ErnieImagePostInfer
from lightx2v.models.networks.ernie_image.infer.pre_infer import ErnieImagePreInfer
from lightx2v.models.networks.ernie_image.infer.transformer_infer import ErnieImageTransformerInfer
from lightx2v.models.networks.ernie_image.weights.post_weights import ErnieImagePostWeights
from lightx2v.models.networks.ernie_image.weights.pre_weights import ErnieImagePreWeights
from lightx2v.models.networks.ernie_image.weights.transformer_weights import ErnieImageTransformerWeights
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class ErnieImageTransformerModel(BaseTransformerModel):
    pre_weight_class = ErnieImagePreWeights
    transformer_weight_class = ErnieImageTransformerWeights
    post_weight_class = ErnieImagePostWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, None, lora_path, lora_strength)
        if self.config.get("seq_parallel", False):
            raise NotImplementedError("ERNIE-Image native DiT does not support seq_parallel yet.")
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("ERNIE-Image feature caching is not implemented.")
        self.pre_infer_class = ErnieImagePreInfer
        self.transformer_infer_class = ErnieImageTransformerInfer
        self.post_infer_class = ErnieImagePostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, prompt_embeds, infer_condition=True):
        self.scheduler.infer_condition = infer_condition
        pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            hidden_states=latents_input,
            encoder_hidden_states=prompt_embeds,
        )
        hidden_states = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            pre_infer_out=pre_infer_out,
        )
        return self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out)

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        raise NotImplementedError("ERNIE-Image native DiT does not support seq_parallel yet.")

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        raise NotImplementedError("ERNIE-Image native DiT does not support seq_parallel yet.")

    @compiled_method()
    @torch.no_grad()
    def infer(self, inputs):
        if self.config.get("cfg_parallel", False):
            raise NotImplementedError("ERNIE-Image native DiT does not support cfg_parallel yet.")

        if self.cpu_offload:
            self.to_cuda()

        latents = self.scheduler.latents
        text_output = inputs["text_encoder_output"]

        if self.config.get("enable_cfg", False):
            noise_pred_cond = self._infer_cond_uncond(
                latents,
                text_output["prompt_embeds"],
                infer_condition=True,
            )
            noise_pred_uncond = self._infer_cond_uncond(
                latents,
                text_output["negative_prompt_embeds"],
                infer_condition=False,
            )
            guidance_scale = self.scheduler.sample_guide_scale
            self.scheduler.noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            self.scheduler.noise_pred = self._infer_cond_uncond(
                latents,
                text_output["prompt_embeds"],
                infer_condition=True,
            )

        if self.cpu_offload:
            self.to_cpu()
            torch_device_module.empty_cache()
            gc.collect()
