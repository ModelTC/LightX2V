"""Avoid the empty video projection in Base10's audio-only branch."""

from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.utils.envs import GET_DTYPE


class TaoMateBase10PreInfer(MiniMaxH3PreInfer):
    def prepare_inputs(self, weights, prompt_embeds):
        layout = self.scheduler.layout
        audio = weights.audio_proj_in.apply(self.scheduler.audio_latents.float()).to(GET_DTYPE())
        hidden = audio.new_zeros((layout.sequence_length, self.hidden_size))
        if self.scheduler.refined_text is None:
            text = weights.context_embedder.apply(prompt_embeds.to(GET_DTYPE()))
            self.scheduler.refined_text = self._refine_text(weights, text)
        hidden.index_copy_(0, layout.text_indices, self.scheduler.refined_text)
        hidden.index_copy_(0, layout.audio_indices, audio)
        return hidden, self._rotary_embedding(layout.position_ids)
