import json
import os

import torch

try:
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
except ImportError:
    QwenImageTransformer2DModel = None

from .infer.post_infer import QwenImagePostInfer
from .infer.pre_infer import QwenImagePreInfer
from .infer.transformer_infer import QwenImageTransformerInfer
from .layers.linear import DefaultLinear, replace_linear_with_custom
from .layers.normalization import DefaultLayerNorm, DefaultRMSNorm, replace_layernorm_with_custom, replace_rmsnorm_with_custom


class QwenImageTransformerModel:
    def __init__(self, config):
        self.config = config
        self.transformer = QwenImageTransformer2DModel.from_pretrained(os.path.join(config.model_path, "transformer"))
        # repalce linear & normalization
        self.transformer = replace_linear_with_custom(self.transformer, DefaultLinear)
        self.transformer = replace_layernorm_with_custom(self.transformer, DefaultLayerNorm)
        self.transformer = replace_rmsnorm_with_custom(self.transformer, DefaultRMSNorm)
        self.transformer.to(torch.device("cuda")).to(torch.bfloat16)

        with open(os.path.join(config.model_path, "transformer", "config.json"), "r") as f:
            transformer_config = json.load(f)
            self.in_channels = transformer_config["in_channels"]
        self.attention_kwargs = {}

        self._init_infer_class()
        self._init_infer()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _init_infer_class(self):
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = QwenImageTransformerInfer
        else:
            assert NotImplementedError
        self.pre_infer_class = QwenImagePreInfer
        self.post_infer_class = QwenImagePostInfer

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config, self.transformer.transformer_blocks)
        self.pre_infer = self.pre_infer_class(self.config, self.transformer.img_in, self.transformer.txt_norm, self.transformer.txt_in, self.transformer.time_text_embed, self.transformer.pos_embed)
        self.post_infer = self.post_infer_class(self.config, self.transformer.norm_out, self.transformer.proj_out)

    @torch.no_grad()
    def infer(self, inputs):
        t = self.scheduler.timesteps[self.scheduler.step_index]
        latents = self.scheduler.latents
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        img_shapes = self.scheduler.img_shapes

        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]
        prompt_embeds_mask = inputs["text_encoder_output"]["prompt_embeds_mask"]

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None

        hidden_states, encoder_hidden_states, encoder_hidden_states_mask, pre_infer_out = self.pre_infer.infer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=self.scheduler.guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=self.attention_kwargs,
        )

        encoder_hidden_states, hidden_states = self.transformer_infer.infer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pre_infer_out=pre_infer_out,
            attention_kwargs=self.attention_kwargs,
        )

        noise_pred = self.post_infer.infer(hidden_states, pre_infer_out[1])

        self.scheduler.noise_pred = noise_pred
