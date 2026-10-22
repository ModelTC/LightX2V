"""Request lifecycle shared by H3 autoregressive runners."""

import torch


class H3AutoregressiveRunnerMixin:
    """Release request-local H3 state while retaining reusable model weights."""

    def run_request(self, input_info):
        try:
            return super().run_request(input_info)
        finally:
            self.model.reset_autoregressive_state()

    def close(self):
        if self.model is not None:
            self.model.reset_cuda_graph()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        super().close()
