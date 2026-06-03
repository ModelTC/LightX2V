import os

import torch

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.hidream_o1_image.infer.post_infer import HidreamO1ImagePostInfer
from lightx2v.models.networks.hidream_o1_image.infer.pre_infer import HidreamO1ImagePreInfer
from lightx2v.models.networks.hidream_o1_image.infer.transformer_infer import HidreamO1ImageTransformerInfer
from lightx2v.models.networks.hidream_o1_image.qwen3_vl import Qwen3VLConfig, Qwen3VLTextRotaryEmbedding
from lightx2v.models.networks.hidream_o1_image.weights.post_weights import HidreamO1ImagePostWeights
from lightx2v.models.networks.hidream_o1_image.weights.pre_weights import HidreamO1ImagePreWeights
from lightx2v.models.networks.hidream_o1_image.weights.transformer_weights import HidreamO1ImageTransformerWeights
from lightx2v_platform.base.global_var import AI_DEVICE


class HidreamO1ImageModel(BaseTransformerModel):
    pre_weight_class = HidreamO1ImagePreWeights
    transformer_weight_class = HidreamO1ImageTransformerWeights
    post_weight_class = HidreamO1ImagePostWeights

    def __init__(self, model_path, config, torch_dtype):
        self.dtype = torch_dtype
        self.qwen_config = Qwen3VLConfig.from_json_file(os.path.join(model_path, "config.json"))
        self._populate_model_config(config, self.qwen_config)
        config["_hidream_model_config"] = self.qwen_config
        self.preserved_keys = [
            "model.language_model",
            "model.t_embedder1",
            "model.x_embedder",
            "model.final_layer2",
            "model.visual",
        ]
        super().__init__(model_path, config, torch.device(AI_DEVICE), None)
        self._init_infer_class()
        self._init_weights()
        self._configure_weight_structure()
        self._init_infer()
        self._cached_image_embeds = None
        self._cached_deepstack_image_embeds = None

    def _populate_model_config(self, config, qwen_config):
        text_config = qwen_config.text_config
        config["num_hidden_layers"] = text_config.num_hidden_layers
        config["num_layers"] = text_config.num_hidden_layers
        config["rms_norm_eps"] = text_config.rms_norm_eps
        config["tms_token_id"] = 151673
        config["timestep_frequency_embedding_size"] = 256

    @property
    def model_config(self):
        return self.qwen_config

    def _configure_weight_structure(self):
        rotary_emb = Qwen3VLTextRotaryEmbedding(config=self.qwen_config.text_config).to(self.device)
        self.pre_weight.configure_model(self.qwen_config)
        self.transformer_weights.configure_model(self.qwen_config.text_config, rotary_emb, self.device)

    def _init_infer_class(self):
        self.pre_infer_class = HidreamO1ImagePreInfer
        self.transformer_infer_class = HidreamO1ImageTransformerInfer
        self.post_infer_class = HidreamO1ImagePostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    @torch.no_grad()
    def _infer_cond_uncond(self, sample, z_in, t_pixeldit, precomputed_image_embeds=None, precomputed_deepstack_image_embeds=None):
        pre_out = self.pre_infer.infer(
            self.pre_weight,
            sample,
            z_in,
            t_pixeldit,
            precomputed_image_embeds=precomputed_image_embeds,
            precomputed_deepstack_image_embeds=precomputed_deepstack_image_embeds,
        )
        transformer_out = self.transformer_infer.infer(self.transformer_weights, pre_out, self.dtype)
        x_pred = self.post_infer.infer(self.post_weight, transformer_out)
        return x_pred, pre_out.cond_image_embeds, pre_out.cond_deepstack_image_embeds

    @torch.no_grad()
    def infer(self, inputs):
        cfg = inputs["generation_config"]
        samples = inputs["samples"]
        latents = self.scheduler.latents
        t_pixeldit = self.scheduler.current_t_pixeldit
        sigma = self.scheduler.current_sigma

        if self.scheduler.step_index == 0:
            self._cached_image_embeds = None
            self._cached_deepstack_image_embeds = None

        z_in = latents.clone()
        if "ref_patches" in inputs:
            z_in = torch.cat([z_in, inputs["ref_patches"].to(latents.device, latents.dtype)], dim=1)

        x_pred_cond, image_embeds, deepstack_image_embeds = self._infer_cond_uncond(
            samples[0],
            z_in,
            t_pixeldit,
            precomputed_image_embeds=self._cached_image_embeds,
            precomputed_deepstack_image_embeds=self._cached_deepstack_image_embeds,
        )
        if image_embeds is not None and deepstack_image_embeds is not None and self._cached_image_embeds is None:
            self._cached_image_embeds = image_embeds.detach()
            self._cached_deepstack_image_embeds = [item.detach() for item in deepstack_image_embeds]

        v_cond = (x_pred_cond.to(dtype=torch.float32) - latents.to(dtype=torch.float32)) / sigma
        if len(samples) > 1:
            x_pred_uncond, image_embeds, deepstack_image_embeds = self._infer_cond_uncond(
                samples[1],
                z_in,
                t_pixeldit,
                precomputed_image_embeds=self._cached_image_embeds,
                precomputed_deepstack_image_embeds=self._cached_deepstack_image_embeds,
            )
            if image_embeds is not None and deepstack_image_embeds is not None and self._cached_image_embeds is None:
                self._cached_image_embeds = image_embeds.detach()
                self._cached_deepstack_image_embeds = [item.detach() for item in deepstack_image_embeds]
            v_uncond = (x_pred_uncond.to(dtype=torch.float32) - latents.to(dtype=torch.float32)) / sigma
            v_guided = v_uncond + cfg["guidance_scale"] * (v_cond - v_uncond)
        else:
            v_guided = v_cond
        self.scheduler.noise_pred = -v_guided

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        return x
