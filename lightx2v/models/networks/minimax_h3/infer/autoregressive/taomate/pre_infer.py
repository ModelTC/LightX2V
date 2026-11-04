"""TaoMate-H3 packed-row metadata for the shared H3 pre-infer stage."""

from dataclasses import dataclass

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3PreInferOutput
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer


@dataclass
class TaoMateH3PreInferOutput(MiniMaxH3PreInferOutput):
    valid_sequence_length: int | None = None
    attention_state: object | None = None


class TaoMateH3PreInfer(MiniMaxH3PreInfer):
    def infer(self, weights, prompt_embeds):
        output = super().infer(weights, prompt_embeds)
        return TaoMateH3PreInferOutput(**vars(output), valid_sequence_length=self.scheduler.layout.sequence_length)


__all__ = ["TaoMateH3PreInfer"]
