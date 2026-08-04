import os

import torch
from safetensors import safe_open

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.networks.minimax_h3.weights import (
    MiniMaxH3PostWeights,
    MiniMaxH3PreWeights,
    MiniMaxH3TransformerWeights,
)
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3Model(BaseTransformerModel):
    """LightX2V-native MiniMax-H3 joint audio/video transformer."""

    pre_weight_class = MiniMaxH3PreWeights
    transformer_weight_class = MiniMaxH3TransformerWeights
    post_weight_class = MiniMaxH3PostWeights

    def __init__(self, model_path, config, device):
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError(
                "MiniMax-H3 requires DTYPE=BF16. The native loader preserves the released checkpoint's 626 BF16 tensors and 12 FP32 projection/time/head tensors without dtype conversion."
            )
        if config.get("seq_parallel", False):
            raise NotImplementedError("MiniMax-H3 sequence parallel support is not implemented yet")
        if config.get("tensor_parallel", False):
            raise NotImplementedError("MiniMax-H3 tensor parallel support is not implemented yet")
        if config.get("cfg_parallel", False) or config.get("enable_cfg", False):
            raise ValueError("MiniMax-H3 is guidance-distilled and does not have a CFG/unconditional branch")
        if config.get("dit_quantized", False):
            raise NotImplementedError("MiniMax-H3 currently supports the released mixed BF16/FP32 weights")
        if config.get("lora_configs"):
            raise NotImplementedError("MiniMax-H3 LoRA loading is not implemented yet")
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") != "model":
            raise NotImplementedError("MiniMax-H3 currently supports component/model CPU offload; block offload needs block-sharded weights")

        transformer_path = config.get("dit_original_ckpt") or os.path.join(model_path, "transformer")
        super().__init__(transformer_path, config, device)
        self.sensitive_layer = {
            "proj_in",
            "audio_proj_in",
            "time_embedder",
            "proj_out",
            "audio_proj_out",
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        """Load the released mixed-precision tensors without generic dtype coercion."""
        del unified_dtype, sensitive_layer
        if os.path.splitext(file_path)[-1] != ".safetensors":
            raise ValueError(f"MiniMax-H3 native loading expects the released safetensors checkpoint; got {file_path}")
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
        preserve_keys = self.preserved_keys if hasattr(self, "preserved_keys") else None
        with safe_open(file_path, framework="pt", device=str(self.device)) as source:
            return {
                key: source.get_tensor(key)
                for key in source.keys()
                if not any(remove_key in key for remove_key in remove_keys) and (preserve_keys is None or any(preserve_key in key for preserve_key in preserve_keys))
            }

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("MiniMax-H3 feature caching is not implemented")
        self.pre_infer_class = MiniMaxH3PreInfer
        self.transformer_infer_class = MiniMaxH3TransformerInfer
        self.post_infer_class = MiniMaxH3PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        if not infer_condition:
            raise ValueError("MiniMax-H3 does not execute an unconditional pass")
        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]
        pre = self.pre_infer.infer(self.pre_weight, prompt_embeds)
        hidden_states = self.transformer_infer.infer(self.transformer_weights, pre)
        return self.post_infer.infer(self.post_weight, hidden_states, pre)

    @torch.no_grad()
    def infer(self, inputs):
        output = self._infer_cond_uncond(inputs, infer_condition=True)
        self.scheduler.video_noise_pred = output.video
        self.scheduler.audio_noise_pred = output.audio

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        raise NotImplementedError("MiniMax-H3 sequence parallel support is not implemented")

    @torch.no_grad()
    def _seq_parallel_post_process(self, output):
        raise NotImplementedError("MiniMax-H3 sequence parallel support is not implemented")
