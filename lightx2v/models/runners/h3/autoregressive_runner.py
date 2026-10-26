"""Request lifecycle shared by H3 autoregressive runners."""

from lightx2v.models.runners.autoregressive_runner import AutoregressiveRunnerMixin


class H3AutoregressiveRunnerMixin(AutoregressiveRunnerMixin):
    """Release request-local H3 state while retaining reusable model weights."""

    def run_request(self, input_info):
        try:
            return super().run_request(input_info)
        finally:
            self.model.reset_autoregressive_state()

    def close(self):
        super().close()
        # Graphs and device transfers are retired by the common close before
        # unmapping shared host storage. A video VAE owns its arena separately.
        for component in (getattr(self, "model", None), getattr(self, "video_vae", None)):
            close_weights = getattr(component, "close_shared_cpu_weights", None)
            if close_weights is not None:
                manager = getattr(getattr(component, "transformer_infer", None), "offload_manager", None)
                close_manager = getattr(manager, "close", None)
                if close_manager is not None:
                    close_manager()
                close_weights()
