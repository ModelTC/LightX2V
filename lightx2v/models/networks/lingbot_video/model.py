import torch

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.lingbot_video.infer.post_infer import LingBotVideoPostInfer
from lightx2v.models.networks.lingbot_video.infer.pre_infer import LingBotVideoPreInfer
from lightx2v.models.networks.lingbot_video.infer.transformer_infer import LingBotVideoTransformerInfer
from lightx2v.models.networks.lingbot_video.weights.post_weights import LingBotVideoPostWeights
from lightx2v.models.networks.lingbot_video.weights.pre_weights import LingBotVideoPreWeights
from lightx2v.models.networks.lingbot_video.weights.transformer_weights import LingBotVideoTransformerWeights
from lightx2v.utils.custom_compiler import compiled_method


class LingBotVideoTransformerModel(BaseTransformerModel):
    pre_weight_class = LingBotVideoPreWeights
    transformer_weight_class = LingBotVideoTransformerWeights
    post_weight_class = LingBotVideoPostWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        if config.get("seq_parallel", False):
            raise NotImplementedError("LingBot-Video currently supports single-device serial inference only.")
        if config.get("cfg_parallel", False):
            raise NotImplementedError("LingBot-Video uses serial CFG in this LightX2V backend.")
        if config.get("cpu_offload", False):
            raise NotImplementedError("LingBot-Video CPU offload is not implemented yet.")
        super().__init__(model_path, config, device, None, lora_path, lora_strength)
        self.sensitive_layer = {
            "time_embedder",
            "time_modulation",
            "scale_shift_table",
            "norm",
            "norm1",
            "norm2",
            "norm_q",
            "norm_k",
            "norm_post_attn",
            "norm_post_ffn",
            "norm_out",
            "norm_out_modulation",
            "router",
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("LingBot-Video feature caching is not implemented.")
        self.pre_infer_class = LingBotVideoPreInfer
        self.transformer_infer_class = LingBotVideoTransformerInfer
        self.post_infer_class = LingBotVideoPostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    def _load_lora_file(self, file_path):
        lora_weights = super()._load_lora_file(file_path)
        return {key.replace(".lora.down.weight", ".lora_down.weight").replace(".lora.up.weight", ".lora_up.weight"): value for key, value in lora_weights.items()}

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
        raise NotImplementedError("LingBot-Video sequence parallel inference is not implemented.")

    @torch.no_grad()
    def _seq_parallel_post_process(self, noise_pred):
        raise NotImplementedError("LingBot-Video sequence parallel inference is not implemented.")

    @compiled_method()
    @torch.no_grad()
    def infer(self, inputs):
        latents_input = self.scheduler.latents
        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]

        if self.config.get("enable_cfg", True):
            negative_prompt_embeds = inputs["text_encoder_output"]["negative_prompt_embeds"]
            noise_pred_cond = self._infer_cond_uncond(latents_input, prompt_embeds, infer_condition=True)
            noise_pred_uncond = self._infer_cond_uncond(latents_input, negative_prompt_embeds, infer_condition=False)
            self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            self.scheduler.noise_pred = self._infer_cond_uncond(latents_input, prompt_embeds, infer_condition=True)
