from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer


class WanDancerPostInfer(WanPostInfer):
    def infer(self, x, pre_infer_out):
        # Upstream keeps the Euler trajectory in bf16; WanPostInfer normally casts fp32.
        return self.unpatchify(x, pre_infer_out.grid_sizes.tuple)
