from lightx2v.models.networks.qwen_image_21.infer.transformer_infer import QwenImage21TransformerInfer


class QwenImage21MpsTransformerInfer(QwenImage21TransformerInfer):
    """Use the ordinary forward loops over disk-prefetched WeightModule blocks."""

    def get_compile_block_key(self, block_idx, block):
        return id(block)
