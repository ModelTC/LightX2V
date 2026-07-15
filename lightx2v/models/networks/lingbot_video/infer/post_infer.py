import torch.nn.functional as F

from lightx2v.utils.envs import GET_DTYPE


class LingBotVideoPostInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.out_channels = int(config.get("out_channels", 16))

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, pre_infer_out):
        final_mod = weights.norm_out_modulation.apply(F.silu(pre_infer_out.temb_input.float()))
        shift, scale = final_mod.chunk(2, dim=-1)
        final_hidden = weights.norm_out.apply(hidden_states) * (1.0 + scale) + shift
        projected = weights.proj_out.apply(final_hidden.to(GET_DTYPE()))
        video_tokens = projected[: pre_infer_out.n_video]

        p_f, p_h, p_w = self.patch_size
        grid_t, grid_h, grid_w = pre_infer_out.grid_t, pre_infer_out.grid_h, pre_infer_out.grid_w
        output = video_tokens.reshape(1, grid_t, grid_h, grid_w, p_f, p_h, p_w, self.out_channels)
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            1,
            self.out_channels,
            grid_t * p_f,
            grid_h * p_h,
            grid_w * p_w,
        )
        return output
