from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER, EMBEDDING_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER


class WanPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config["in_dim"]
        self.dim = config["dim"]
        self.patch_size = (1, 2, 2)
        self.config = config

        self.add_module(
            "patch_embedding",
            CONV3D_WEIGHT_REGISTER["Default"](
                "patch_embedding.weight",
                "patch_embedding.bias",
                stride=self.patch_size,
                lora_prefix="diffusion_model.patch_embedding",
            ),
        )
        self.enable_lingbot_cam_ctrl = config.get("enable_lingbot_cam_ctrl", False) or config.get("model_cls") == "wan2.2_moe_lingbot"

        if self.enable_lingbot_cam_ctrl:
            self.add_module(
                "patch_embedding_wancamctrl",
                MM_WEIGHT_REGISTER["Default"](
                    "patch_embedding_wancamctrl.weight",
                    "patch_embedding_wancamctrl.bias",
                    lora_prefix="diffusion_model.patch_embedding_wancamctrl",
                ),
            )
            self.add_module(
                "c2ws_hidden_states_layer1",
                MM_WEIGHT_REGISTER["Default"](
                    "c2ws_hidden_states_layer1.weight",
                    "c2ws_hidden_states_layer1.bias",
                    lora_prefix="diffusion_model.c2ws_hidden_states_layer1",
                ),
            )
            self.add_module(
                "c2ws_hidden_states_layer2",
                MM_WEIGHT_REGISTER["Default"](
                    "c2ws_hidden_states_layer2.weight",
                    "c2ws_hidden_states_layer2.bias",
                    lora_prefix="diffusion_model.c2ws_hidden_states_layer2",
                ),
            )

        if config["task"] in ["rs2v"]:
            self.add_module(
                "ref_patch_embedding",
                CONV3D_WEIGHT_REGISTER["Default"](
                    "ref_patch_embedding.weight",
                    "ref_patch_embedding.bias",
                    stride=self.patch_size,
                    lora_prefix="diffusion_model.ref_patch_embedding",
                ),
            )
            self.add_module(
                "prev_patch_embedding",
                CONV3D_WEIGHT_REGISTER["Default"](
                    "prev_patch_embedding.weight",
                    "prev_patch_embedding.bias",
                    stride=self.patch_size,
                    lora_prefix="diffusion_model.prev_patch_embedding",
                ),
            )
            self.add_module(
                "cont_patch_embedding",
                CONV3D_WEIGHT_REGISTER["Default"](
                    "cont_patch_embedding.weight",
                    "cont_patch_embedding.bias",
                    stride=self.patch_size,
                    lora_prefix="diffusion_model.cont_patch_embedding",
                ),
            )
            self.add_module(
                "state_embedding",
                EMBEDDING_WEIGHT_REGISTER["Default"](
                    "state_embedding.weight",
                ),
            )

        self.add_module(
            "text_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                "text_embedding.0.weight",
                "text_embedding.0.bias",
                lora_prefix="diffusion_model.text_embedding",
            ),
        )
        self.add_module(
            "text_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                "text_embedding.2.weight",
                "text_embedding.2.bias",
                lora_prefix="diffusion_model.text_embedding",
            ),
        )
        self.add_module(
            "time_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                "time_embedding.0.weight",
                "time_embedding.0.bias",
                lora_prefix="diffusion_model.time_embedding",
            ),
        )
        self.add_module(
            "time_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                "time_embedding.2.weight",
                "time_embedding.2.bias",
                lora_prefix="diffusion_model.time_embedding",
            ),
        )
        self.add_module(
            "time_projection_1",
            MM_WEIGHT_REGISTER["Default"](
                "time_projection.1.weight",
                "time_projection.1.bias",
                lora_prefix="diffusion_model.time_projection",
            ),
        )

        if config["task"] in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and config.get("use_image_encoder", True):
            self.add_module(
                "proj_0",
                LN_WEIGHT_REGISTER["torch"](
                    "img_emb.proj.0.weight",
                    "img_emb.proj.0.bias",
                    lora_prefix="diffusion_model.img_emb",
                ),
            )
            self.add_module(
                "proj_1",
                MM_WEIGHT_REGISTER["Default"](
                    "img_emb.proj.1.weight",
                    "img_emb.proj.1.bias",
                    lora_prefix="diffusion_model.img_emb",
                ),
            )
            self.add_module(
                "proj_3",
                MM_WEIGHT_REGISTER["Default"](
                    "img_emb.proj.3.weight",
                    "img_emb.proj.3.bias",
                    lora_prefix="diffusion_model.img_emb",
                ),
            )
            self.add_module(
                "proj_4",
                LN_WEIGHT_REGISTER["torch"](
                    "img_emb.proj.4.weight",
                    "img_emb.proj.4.bias",
                    lora_prefix="diffusion_model.img_emb",
                ),
            )

        if config["model_cls"] == "wan2.1_distill" and config.get("enable_dynamic_cfg", False):
            self.add_module(
                "cfg_cond_proj_1",
                MM_WEIGHT_REGISTER["Default"](
                    "guidance_embedding.linear_1.weight",
                    "guidance_embedding.linear_1.bias",
                ),
            )
            self.add_module(
                "cfg_cond_proj_2",
                MM_WEIGHT_REGISTER["Default"](
                    "guidance_embedding.linear_2.weight",
                    "guidance_embedding.linear_2.bias",
                ),
            )

        if config["model_cls"] == "wan2.1_mean_flow_distill":
            self.add_module(
                "time_embedding_r_0",
                MM_WEIGHT_REGISTER["Default"]("time_embedding_r.0.weight", "time_embedding_r.0.bias"),
            )
            self.add_module(
                "time_embedding_r_2",
                MM_WEIGHT_REGISTER["Default"]("time_embedding_r.2.weight", "time_embedding_r.2.bias"),
            )

        if config["task"] == "flf2v" and config.get("use_image_encoder", True):
            self.add_module(
                "emb_pos",
                TENSOR_REGISTER["Default"](f"img_emb.emb_pos"),
            )
        if config["task"] == "animate":
            self.add_module(
                "pose_patch_embedding",
                CONV3D_WEIGHT_REGISTER["Default"](
                    "pose_patch_embedding.weight",
                    "pose_patch_embedding.bias",
                    stride=self.patch_size,
                ),
            )
