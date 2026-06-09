class HidreamO1ImagePostInfer:
    def __init__(self, config):
        self.config = config

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, transformer_infer_out):
        x_vis = transformer_infer_out.hidden_states[0]
        if transformer_infer_out.tgt_image_len is not None:
            x_vis = x_vis[: transformer_infer_out.tgt_image_len]
        return x_vis.unsqueeze(0)
