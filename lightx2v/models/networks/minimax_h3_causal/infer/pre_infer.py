from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.models.networks.minimax_h3_causal.infer.module_io import MiniMaxH3CausalPreInferOutput


class MiniMaxH3CausalPreInfer(MiniMaxH3PreInfer):
    def prepare_inputs(self, weights, prompt_embeds):
        scheduler = self.scheduler
        if scheduler.prepared_inputs is None:
            hidden_states, rotary_emb = super().prepare_inputs(weights, prompt_embeds)
            key_rotary_emb = self._rotary_embedding(scheduler.key_position_ids)
            scheduler.prepared_inputs = hidden_states, rotary_emb, key_rotary_emb
            return hidden_states.clone(), rotary_emb

        # Text/reference inputs stay fixed during prefill; only video changes
        # between media denoising steps. Keep the cached template unmodified.
        hidden_states, rotary_emb, _ = scheduler.prepared_inputs
        hidden_states = hidden_states.clone()
        if not scheduler.condition_phase:
            video_embeds = weights.proj_in.apply(scheduler.video_latents.float()).to(hidden_states.dtype)
            hidden_states.index_copy_(0, scheduler.layout.video_indices, video_embeds)
        return hidden_states, rotary_emb

    def infer(self, weights, prompt_embeds):
        output = super().infer(weights, prompt_embeds)
        return MiniMaxH3CausalPreInferOutput(
            **vars(output),
            key_rotary_emb=self.scheduler.prepared_inputs[2],
            valid_sequence_length=self.scheduler.layout.sequence_length,
        )
