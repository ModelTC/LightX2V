import torch

from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3ImagePreInfer(MiniMaxH3PreInfer):
    def prepare_inputs(self, weights, prompt_embeds):
        layout = self.scheduler.layout
        text_embeds = weights.context_embedder.apply(prompt_embeds.to(GET_DTYPE()))
        text_embeds = self._refine_text(weights, text_embeds)
        image_embeds = weights.proj_in.apply(self.scheduler.video_latents.float()).to(GET_DTYPE())
        hidden_states = torch.cat((text_embeds, image_embeds), dim=0)
        return hidden_states, self._rotary_embedding(layout.position_ids)


class MiniMaxH3ImagePostInfer(MiniMaxH3PostInfer):
    def infer(self, weights, hidden_states, pre_infer_out):
        # Each PDD step uses its own FP32 output projection.
        hidden_states = self.norm_out(weights, hidden_states, pre_infer_out)
        head = weights.pdd_heads[self.scheduler.step_index].proj
        return head.apply(hidden_states.float()).index_select(0, pre_infer_out.video_indices)
