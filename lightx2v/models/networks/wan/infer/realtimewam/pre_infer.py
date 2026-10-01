import torch

from lightx2v.models.networks.wan.infer.fastwam.pre_infer import FastWAMPreInfer


class RealtimeWAMPreInfer(FastWAMPreInfer):
    def infer_video(self, pre_weight, first_frame_latents, context, context_mask):
        pre = super().infer_video(pre_weight, first_frame_latents, context, context_mask)
        if self.config.get("kv_fusion", False):
            times = torch.full((len(pre.tokens),), 1000.0, device=pre.tokens.device, dtype=pre.tokens.dtype)
            times[: pre.tokens_per_frame] = 0
            pre.t, pre.t_mod = self._time_embedding(pre_weight.video, times, self.video_hidden_dim)
        return pre
