import torch

from lightx2v.models.networks.wan.infer.animate2 import (
    WanAnimate2PreInfer,
    WanAnimate2TransformerInfer,
)
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.model import WanModel


class WanAnimate2Model(WanModel):
    """Native LightX2V implementation of the Wan-Animate-2 DiT."""

    def __init__(
        self,
        model_path,
        config,
        device,
        model_type="wan2.1",
        lora_path=None,
        lora_strength=1.0,
    ):
        super().__init__(
            model_path,
            config,
            device,
            model_type=model_type,
            lora_path=lora_path,
            lora_strength=lora_strength,
        )

    @staticmethod
    def _normalize_checkpoint_keys(weight_dict):
        """Remove the released checkpoint's Incontext_AttentionBlock wrapper."""
        normalized = {}
        for key, value in weight_dict.items():
            parts = key.split(".")
            if len(parts) >= 4 and parts[0] == "blocks" and parts[1].isdigit() and parts[2] == "block":
                key = ".".join(parts[:2] + parts[3:])
            if key in normalized:
                raise ValueError(f"Wan-Animate-2 checkpoint key collision after normalization: {key}")
            normalized[key] = value
        return normalized

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        return self._normalize_checkpoint_keys(super()._load_ckpt(unified_dtype, sensitive_layer))

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        return self._normalize_checkpoint_keys(super()._load_quant_ckpt(unified_dtype, sensitive_layer))

    def _load_dummy_ckpt(self, unified_dtype, sensitive_layer):
        return self._normalize_checkpoint_keys(super()._load_dummy_ckpt(unified_dtype, sensitive_layer))

    def _init_infer_class(self):
        self.pre_infer_class = WanAnimate2PreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanAnimate2TransformerInfer

    @torch.no_grad()
    def prepare_reference(self, inputs):
        """Prefill every layer's immutable driving-reference K/V once per clip."""
        if self.cpu_offload:
            if self.offload_granularity == "model":
                self.to_cuda()
            else:
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()
        try:
            pre_infer_out = self.pre_infer.infer_reference(self.pre_weight, inputs)
            if self.config["seq_parallel"]:
                pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)
            self.transformer_infer.infer_reference(self.transformer_weights, pre_infer_out)
        finally:
            # Resident runners must restore their requested offload state even
            # when a prefill is cancelled or raises partway through a block.
            if self.cpu_offload:
                if self.offload_granularity == "model":
                    self.to_cpu()
                else:
                    self.pre_weight.to_cpu()
                    self.transformer_weights.non_block_weights_to_cpu()
