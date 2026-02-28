from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, TENSOR_REGISTER


class SeedVRPostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.layer_norm_type = config.get("layer_norm_type", "torch")
        self.norm_eps = config.get("norm_eps", 1.0e-5)

        vid_out_norm = config.get("vid_out_norm", None)
        if vid_out_norm is not None:
            if vid_out_norm in ["rms", "fusedrms"]:
                self.add_module(
                    "vid_out_norm",
                    RMS_WEIGHT_REGISTER[self.rms_norm_type](
                        "vid_out_norm.weight",
                        eps=self.norm_eps,
                    ),
                )
            elif vid_out_norm in ["layer", "fusedln"]:
                self.add_module(
                    "vid_out_norm",
                    LN_WEIGHT_REGISTER[self.layer_norm_type](
                        "vid_out_norm.weight",
                        "vid_out_norm.bias",
                        eps=self.norm_eps,
                    ),
                )
            else:
                raise NotImplementedError(f"Unsupported vid_out_norm type: {vid_out_norm}")

            self.add_module("vid_out_ada_out_shift", TENSOR_REGISTER["Default"]("vid_out_ada.out_shift"))
            self.add_module("vid_out_ada_out_scale", TENSOR_REGISTER["Default"]("vid_out_ada.out_scale"))

        self.add_module(
            "vid_out_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                "vid_out.proj.weight",
                "vid_out.proj.bias",
            ),
        )
