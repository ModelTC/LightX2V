import json
import os
import torch
import torch.nn as nn
from typing import Type
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from .layers.linear import replace_linear_with_custom, DefaultLinear
from .layers.normalization import replace_layernorm_with_custom, replace_rmsnorm_with_custom, DefaultLayerNorm, DefaultRMSNorm


class QwenImageTransformerModel:
    def __init__(self, config):
        self.config = config
        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            os.path.join(config.model_path, "transformer")
        )
        # repalce linear & normalization
        self.transformer = replace_linear_with_custom(self.transformer, DefaultLinear)
        self.transformer = replace_layernorm_with_custom(self.transformer, DefaultLayerNorm)
        self.transformer = replace_rmsnorm_with_custom(self.transformer, DefaultRMSNorm)
        self.transformer.to(torch.device("cuda")).to(torch.bfloat16)
        breakpoint()
        
        with open(os.path.join(config.model_path, "transformer", "config.json"), "r") as f:
            transformer_config = json.load(f)
            self.in_channels = transformer_config["in_channels"]
        self.attention_kwargs = {}
        
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    @torch.no_grad()
    def infer(self, inputs):
        t = self.scheduler.timesteps[self.scheduler.step_index]
        latents = self.scheduler.latents
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        img_shapes = self.scheduler.img_shapes
        
        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]
        prompt_embeds_mask = inputs["text_encoder_output"]["prompt_embeds_mask"]
        
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None

        with self.transformer.cache_context("cond"):
            noise_pred = self.transformer(
                        hidden_states=latents,  
                        timestep=timestep / 1000, 
                        guidance=self.scheduler.guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds, 
                        img_shapes=img_shapes, 
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
        self.scheduler.noise_pred = noise_pred
        
