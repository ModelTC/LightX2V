"""Stage3 projects video velocity; Base10 supplies every audio milestone."""

from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer


class TaoMateH3PostInfer(MiniMaxH3PostInfer):
    def _project_audio(self, weights, hidden_states, pre_infer_out):
        return hidden_states.new_zeros((pre_infer_out.audio_indices.numel(), self.scheduler.audio_latents.shape[-1]))


__all__ = ["TaoMateH3PostInfer"]
