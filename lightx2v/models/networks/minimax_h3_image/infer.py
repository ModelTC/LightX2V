import torch

from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3ImagePreInfer(MiniMaxH3PreInfer):
    def prepare_inputs(self, weights, prompt_embeds):
        scheduler = self.scheduler
        if scheduler.refined_text_embeds is None:
            # Text conditioning and token positions are fixed for all six PDD steps.
            text_embeds = weights.context_embedder.apply(prompt_embeds.to(GET_DTYPE()))
            scheduler.refined_text_embeds = self._refine_text(weights, text_embeds)
            scheduler.rotary_emb = self._rotary_embedding(scheduler.layout.position_ids)
        image_embeds = weights.proj_in.apply(scheduler.video_latents.float()).to(GET_DTYPE())
        return torch.cat((scheduler.refined_text_embeds, image_embeds), dim=0), scheduler.rotary_emb


class MiniMaxH3ImagePostInfer(MiniMaxH3PostInfer):
    def infer(self, weights, hidden_states, pre_infer_out):
        # Each PDD step uses its own FP32 output projection.
        hidden_states = self.norm_out(weights, hidden_states, pre_infer_out)
        head = weights.pdd_heads[self.scheduler.step_index].proj
        return head.apply(hidden_states.float()).index_select(0, pre_infer_out.video_indices)
