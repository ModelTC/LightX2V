class RunnerLifecycleMixin:
    """Share warmup shape ordering and model request-state cleanup."""

    def get_warmup_resolutions(self):
        target = tuple(self.config["size"])
        return tuple(size for size in self._WARMUP_RESOLUTIONS if size != target) + (target,)

    def reset_model_request_state(self):
        model = getattr(self, "model", None)
        if model is None:
            return
        model.transformer_infer.clear_request_state()
        pre_infer = getattr(model, "pre_infer", None)
        clear_pre_infer = getattr(pre_infer, "clear_request_state", None)
        if clear_pre_infer is not None:
            clear_pre_infer()
