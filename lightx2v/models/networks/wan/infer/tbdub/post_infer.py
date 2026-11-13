from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer


class TBDubPostInfer(WanPostInfer):
    def infer(self, x, pre_infer_out):
        target_tokens = pre_infer_out.adapter_args["target_tokens"]
        # Keep the student prediction dtype: upstream Euler updates are BF16.
        return self.unpatchify(x[target_tokens : 2 * target_tokens], pre_infer_out.grid_sizes.tuple)
