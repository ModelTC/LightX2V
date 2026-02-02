from lightx2v.models.networks.hunyuan_video.infer.post_infer import HunyuanVideo15PostInfer


class WorldPlayPostInfer(HunyuanVideo15PostInfer):
    """
    Post-inference module for WorldPlay model.

    Inherits from HunyuanVideo15PostInfer without modifications.
    """

    def __init__(self, config):
        super().__init__(config)
