import torch

from lightx2v_platform.base.global_var import AI_DEVICE


class BagelPostInfer:
    def __init__(self, config, llm_config):
        self.config = config
        self.use_moe = "Mo" in llm_config["layer_module"]

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, packed_query_sequence, packed_text_indexes=None, packed_vae_token_indexes=None, mode="und"):
        if self.use_moe:
            if mode == "und":
                packed_query_sequence = weights.norm.apply(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = weights.norm.apply(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = weights.norm_moe_gen.apply(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = weights.norm.apply(packed_query_sequence)

        return packed_query_sequence

    def llm2vae(self, weights, x):
        x = x.to(AI_DEVICE).to(torch.bfloat16)
        x = weights.llm2vae.apply(x)
        return x
