class HidreamO1ImagePostInfer:
    def __init__(self, config):
        self.config = config

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, transformer_infer_out):
        x_pred = weights.final_linear.apply(transformer_infer_out.hidden_states.reshape(-1, transformer_infer_out.hidden_states.shape[-1]))
        x_pred = x_pred.reshape(*transformer_infer_out.hidden_states.shape[:-1], -1)
        x_vis = x_pred[0, transformer_infer_out.vinput_mask[0]]
        if transformer_infer_out.tgt_image_len is not None:
            x_vis = x_vis[: transformer_infer_out.tgt_image_len]
        return x_vis.unsqueeze(0)
