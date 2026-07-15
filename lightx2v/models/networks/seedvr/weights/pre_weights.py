from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class SeedVRPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Text input projection
        self.add_module(
            "txt_in",
            MM_WEIGHT_REGISTER["Default"](
                "txt_in.weight",
                "txt_in.bias",
            ),
        )

        # Time embedding MLP
        self.add_module(
            "emb_in_proj_in",
            MM_WEIGHT_REGISTER["Default"](
                "emb_in.proj_in.weight",
                "emb_in.proj_in.bias",
            ),
        )
        self.add_module(
            "emb_in_proj_hid",
            MM_WEIGHT_REGISTER["Default"](
                "emb_in.proj_hid.weight",
                "emb_in.proj_hid.bias",
            ),
        )
        self.add_module(
            "emb_in_proj_out",
            MM_WEIGHT_REGISTER["Default"](
                "emb_in.proj_out.weight",
                "emb_in.proj_out.bias",
            ),
        )

        # Video patch in projection (NaPatchIn.proj)
        if config.get("seedvr_has_vid_in", True):
            self.add_module(
                "vid_in_proj",
                MM_WEIGHT_REGISTER["Default"](
                    "vid_in.proj.weight",
                    "vid_in.proj.bias",
                ),
            )
