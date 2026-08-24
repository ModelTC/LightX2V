from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer


class SwiftVRTransformerInfer(WanTransformerInfer):
    """Wan transformer inference with alternating SwiftVR window layouts."""

    def run_block(self, block_idx, block, x, pre_infer_out):
        window_layouts = pre_infer_out.conditional_dict["window_layouts"]
        window_layout = window_layouts[block_idx % 2]
        return super().run_block(block_idx, block, x, pre_infer_out, {"window_layout": window_layout})
