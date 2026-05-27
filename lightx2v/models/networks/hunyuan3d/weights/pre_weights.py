from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class Hunyuan3DPreWeights(WeightModule):
    """Pre-processing weights for Hunyuan3D shape DiT."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        mm_type = config.get("dit_quant_scheme", "Default")

        self.add_module(
            "x_embedder",
            MM_WEIGHT_REGISTER[mm_type](
                "x_embedder.weight",
                "x_embedder.bias",
            ),
        )
        self.add_module(
            "t_embedder_mlp_0",
            MM_WEIGHT_REGISTER[mm_type](
                "t_embedder.mlp.0.weight",
                "t_embedder.mlp.0.bias",
            ),
        )
        self.add_module(
            "t_embedder_mlp_2",
            MM_WEIGHT_REGISTER[mm_type](
                "t_embedder.mlp.2.weight",
                "t_embedder.mlp.2.bias",
            ),
        )

        guidance_cond_proj_dim = config.get("guidance_cond_proj_dim")
        if guidance_cond_proj_dim is not None:
            self.add_module(
                "t_embedder_cond_proj",
                MM_WEIGHT_REGISTER[mm_type](
                    "t_embedder.cond_proj.weight",
                    None,
                ),
            )
        else:
            self.t_embedder_cond_proj = None
