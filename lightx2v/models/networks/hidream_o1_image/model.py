import os

import torch
import torch.distributed as dist
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.hidream_o1_image.infer.post_infer import HidreamO1ImagePostInfer
from lightx2v.models.networks.hidream_o1_image.infer.pre_infer import HidreamO1ImagePreInfer
from lightx2v.models.networks.hidream_o1_image.infer.transformer_infer import HidreamO1ImageTransformerInfer
from lightx2v.models.networks.hidream_o1_image.qwen3_vl import Qwen3VLTextRotaryEmbedding
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

    def _cache_condition_image_embeds(self, image_embeds, deepstack_image_embeds):
        if image_embeds is not None and deepstack_image_embeds is not None and self._cached_image_embeds is None:
            self._cached_image_embeds = image_embeds.detach()
            self._cached_deepstack_image_embeds = [item.detach() for item in deepstack_image_embeds]

    def _infer_velocity(self, sample, z_in, t_pixeldit, latents, sigma):
        x_pred, image_embeds, deepstack_image_embeds = self._infer_cond_uncond(
            sample,
            z_in,
            t_pixeldit,
            precomputed_image_embeds=self._cached_image_embeds,
            precomputed_deepstack_image_embeds=self._cached_deepstack_image_embeds,
        )
        self._cache_condition_image_embeds(image_embeds, deepstack_image_embeds)
        return (x_pred.to(dtype=torch.float32) - latents.to(dtype=torch.float32)) / sigma

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

        z_in = latents
        if "ref_patches" in inputs:
            z_in = torch.cat([z_in, inputs["ref_patches"].to(latents.device, latents.dtype)], dim=1)

        if cfg["enable_cfg"]:
            if len(samples) < 2:
                raise ValueError("HiDream enable_cfg=True requires both conditional and unconditional samples.")

            if self.config.get("cfg_parallel", False):
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    v_pred = self._infer_velocity(samples[0], z_in, t_pixeldit, latents, sigma)
                else:
                    v_pred = self._infer_velocity(samples[1], z_in, t_pixeldit, latents, sigma)

                v_pred_list = [torch.zeros_like(v_pred) for _ in range(2)]
                dist.all_gather(v_pred_list, v_pred, group=cfg_p_group)
                v_cond = v_pred_list[0]  # cfg_p_rank == 0
                v_uncond = v_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                v_cond = self._infer_velocity(samples[0], z_in, t_pixeldit, latents, sigma)
                v_uncond = self._infer_velocity(samples[1], z_in, t_pixeldit, latents, sigma)

            v_guided = v_uncond + cfg["guidance_scale"] * (v_cond - v_uncond)
        else:
            # ==================== No CFG Processing ====================
            v_cond = self._infer_velocity(samples[0], z_in, t_pixeldit, latents, sigma)
            v_guided = v_cond

        self.scheduler.noise_pred = -v_guided

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        return x
