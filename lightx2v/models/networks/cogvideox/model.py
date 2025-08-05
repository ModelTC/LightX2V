import glob
import json
import math
import os

import torch
from safetensors import safe_open

from lightx2v.models.networks.cogvideox.infer.post_infer import CogvideoxPostInfer
from lightx2v.models.networks.cogvideox.infer.pre_infer import CogvideoxPreInfer
from lightx2v.models.networks.cogvideox.infer.transformer_infer import CogvideoxTransformerInfer
from lightx2v.models.networks.cogvideox.weights.post_weights import CogvideoxPostWeights
from lightx2v.models.networks.cogvideox.weights.pre_weights import CogvideoxPreWeights
from lightx2v.models.networks.cogvideox.weights.transformers_weights import CogvideoxTransformerWeights


class CogvideoxModel:
    pre_weight_class = CogvideoxPreWeights
    post_weight_class = CogvideoxPostWeights
    transformer_weight_class = CogvideoxTransformerWeights

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.pre_infer_class = CogvideoxPreInfer
        self.post_infer_class = CogvideoxPostInfer
        self.transformer_infer_class = CogvideoxTransformerInfer

    def _load_safetensor_to_dict(self, file_path):
        with safe_open(file_path, framework="pt") as f:
            tensor_dict = {key: f.get_tensor(key).to(torch.bfloat16).cuda() for key in f.keys()}
        return tensor_dict

    def _load_ckpt(self):
        safetensors_pattern = os.path.join(self.config.model_path, "transformer", "*.safetensors")
        safetensors_files = glob.glob(safetensors_pattern)

        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in directory: {self.model_path}")
        weight_dict = {}
        for file_path in safetensors_files:
            file_weights = self._load_safetensor_to_dict(file_path)
            weight_dict.update(file_weights)
        return weight_dict

    def _init_weights(self):
        weight_dict = self._load_ckpt()
        with open(os.path.join(self.config.model_path, "transformer", "config.json"), "r") as f:
            transformer_cfg = json.load(f)
        # init weights
        self.pre_weight = self.pre_weight_class(transformer_cfg)
        self.transformer_weights = self.transformer_weight_class(transformer_cfg)
        self.post_weight = self.post_weight_class(transformer_cfg)
        # load weights
        self.pre_weight.load_weights(weight_dict)
        self.transformer_weights.load_weights(weight_dict)
        self.post_weight.load_weights(weight_dict)

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.transformer_infer.set_scheduler(scheduler)

    def to_cpu(self):
        self.pre_weight.to_cpu()
        self.post_weight.to_cpu()
        self.transformer_weights.to_cpu()

    def to_cuda(self):
        self.pre_weight.to_cuda()
        self.post_weight.to_cuda()
        self.transformer_weights.to_cuda()

    @torch.no_grad()
    def infer(self, inputs):
        t = self.scheduler.timesteps[self.scheduler.step_index]
        text_encoder_output = inputs["text_encoder_output"]["context"]
        do_classifier_free_guidance = self.config.guidance_scale > 1.0
        latent_model_input = self.scheduler.latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        timestep = t.expand(latent_model_input.shape[0])

        hidden_states, encoder_hidden_states, emb, infer_shapes = self.pre_infer.infer(
            self.pre_weight,
            latent_model_input[0],
            timestep,
            text_encoder_output[0],
        )

        hidden_states, encoder_hidden_states = self.transformer_infer.infer(
            self.transformer_weights,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=emb,
        )

        noise_pred = self.post_infer.infer(self.post_weight, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=emb, infer_shapes=infer_shapes)

        noise_pred = noise_pred.float()

        if self.config.use_dynamic_cfg:  # True
            self.scheduler.guidance_scale = 1 + self.scheduler.guidance_scale * ((1 - math.cos(math.pi * ((self.scheduler.infer_steps - t.item()) / self.scheduler.infer_steps) ** 5.0)) / 2)

        if do_classifier_free_guidance:  # False
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.scheduler.guidance_scale * (noise_pred_text - noise_pred_uncond)

        self.scheduler.noise_pred = noise_pred
