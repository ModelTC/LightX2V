"""TaoMate-H3 packed-row metadata for the shared H3 pre-infer stage."""

from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer


class TaoMateH3PreInfer(MiniMaxH3PreInfer):
    def infer(self, weights, prompt_embeds):
        output = super().infer(weights, prompt_embeds)
        output.valid_sequence_length = self.scheduler.layout.sequence_length
        return output


__all__ = ["TaoMateH3PreInfer"]
