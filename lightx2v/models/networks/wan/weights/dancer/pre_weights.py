from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER


class WanDancerMusicLayerWeights(WeightModule):
    def __init__(self, layer_id):
        super().__init__()
        prefix = f"music_encoder.{layer_id}"
        self.register_parameter("in_proj_weight", TENSOR_REGISTER["Default"](f"{prefix}.self_attn.in_proj_weight"))
        self.register_parameter("in_proj_bias", TENSOR_REGISTER["Default"](f"{prefix}.self_attn.in_proj_bias"))
        self.add_module("out_proj", MM_WEIGHT_REGISTER["Default"](f"{prefix}.self_attn.out_proj.weight", f"{prefix}.self_attn.out_proj.bias"))
        self.add_module("linear1", MM_WEIGHT_REGISTER["Default"](f"{prefix}.linear1.weight", f"{prefix}.linear1.bias"))
        self.add_module("linear2", MM_WEIGHT_REGISTER["Default"](f"{prefix}.linear2.weight", f"{prefix}.linear2.bias"))
        self.add_module("norm1", LN_WEIGHT_REGISTER["torch"](f"{prefix}.norm1.weight", f"{prefix}.norm1.bias", eps=1e-5))
        self.add_module("norm2", LN_WEIGHT_REGISTER["torch"](f"{prefix}.norm2.weight", f"{prefix}.norm2.bias", eps=1e-5))


class WanDancerPreWeights(WanPreWeights):
    """Wan-I2V pre-weights plus Dancer's second CLIP branch and music encoder."""

    def __init__(self, config):
        super().__init__(config)

        # torch.nn.LayerNorm defaults to 1e-5 in Dancer's CLIP MLPs.
        self.add_module("proj_0", LN_WEIGHT_REGISTER["torch"]("img_emb.proj.0.weight", "img_emb.proj.0.bias", eps=1e-5))
        self.add_module("proj_4", LN_WEIGHT_REGISTER["torch"]("img_emb.proj.4.weight", "img_emb.proj.4.bias", eps=1e-5))

        self.add_module(
            "patch_embedding_global",
            CONV3D_WEIGHT_REGISTER["Default"](
                "patch_embedding_global.weight",
                "patch_embedding_global.bias",
                stride=self.patch_size,
            ),
        )
        for index, kind in ((0, "ln"), (1, "mm"), (3, "mm"), (4, "ln")):
            cls = LN_WEIGHT_REGISTER["torch"] if kind == "ln" else MM_WEIGHT_REGISTER["Default"]
            extra = {"eps": 1e-5} if kind == "ln" else {}
            self.add_module(
                f"ref_proj_{index}",
                cls(f"img_emb_refimage.proj.{index}.weight", f"img_emb_refimage.proj.{index}.bias", **extra),
            )

        self.add_module("music_projection", MM_WEIGHT_REGISTER["Default"]("music_projection.weight", "music_projection.bias"))
        self.music_layers = WeightModuleList([WanDancerMusicLayerWeights(i) for i in range(2)])
        self.add_module("music_layers", self.music_layers)
