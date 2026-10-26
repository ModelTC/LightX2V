from lightx2v.models.networks.wan.infer.autoregressive.causal_transformer_infer import WanCausalTransformerInfer


class WanSFTransformerInfer(WanCausalTransformerInfer):
    def prepare_block_input(self, phase, x, pre):
        if hasattr(phase, "before_proj"):
            return phase.before_proj.apply(x) + pre.x
        return x
