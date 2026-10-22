class AutoregressiveModelMixin:
    """Invalidate projected condition KV before changing model weights."""

    def clear_autoregressive_cache(self):
        infer = getattr(self, "transformer_infer", None)
        if infer is not None:
            infer.reset_cross_attn_cache()

    def to_cpu(self):
        infer = getattr(self, "transformer_infer", None)
        if infer is not None:
            infer.reset_cuda_graph()
        return super().to_cpu()

    def _apply_weights(self, weight_dict=None):
        self.clear_autoregressive_cache()
        return super()._apply_weights(weight_dict)

    def _update_lora(self, lora_path, strength):
        self.clear_autoregressive_cache()
        return super()._update_lora(lora_path, strength)

    def _remove_lora(self):
        self.clear_autoregressive_cache()
        return super()._remove_lora()
