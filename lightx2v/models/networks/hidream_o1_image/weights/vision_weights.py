from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, CONV3D_WEIGHT_REGISTER, EMBEDDING_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class HidreamO1ImageVisionWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_type = config.get("mm_type", config.get("dit_quant_scheme", "Default"))
        self.ln_type = config.get("ln_type", "torch")
        self.attn_type = config.get("hidream_vision_attn_type", config.get("hidream_attn_type", "torch_sdpa"))
        self.blocks = WeightModuleList([])
        self.deepstack_merger_list = WeightModuleList([])

    def configure(self, vision_config):
        if getattr(self, "is_configured", False):
            return
        self.is_configured = True
        self.vision_config = vision_config
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.in_channels = vision_config.in_channels
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_grid_per_side = int(vision_config.num_position_embeddings**0.5)
        self.deepstack_visual_indexes = list(vision_config.deepstack_visual_indexes)

        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.add_module("patch_embed", CONV3D_WEIGHT_REGISTER["Default"]("model.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.bias", stride=kernel))
        self.add_module("pos_embed", EMBEDDING_WEIGHT_REGISTER["Default"]("model.visual.pos_embed.weight"))
        self.blocks = WeightModuleList(HidreamO1ImageVisionBlockWeights(i, vision_config, self.mm_type, self.ln_type, self.attn_type) for i in range(vision_config.depth))
        self.add_module("blocks", self.blocks)
        self.add_module("merger", HidreamO1ImageVisionPatchMergerWeights("model.visual.merger", vision_config, self.mm_type, self.ln_type, False))
        self.deepstack_merger_list = WeightModuleList(
            HidreamO1ImageVisionPatchMergerWeights(
                f"model.visual.deepstack_merger_list.{i}",
                vision_config,
                self.mm_type,
                self.ln_type,
                True,
            )
            for i in range(len(self.deepstack_visual_indexes))
        )
        self.add_module("deepstack_merger_list", self.deepstack_merger_list)


class HidreamO1ImageVisionBlockWeights(WeightModule):
    def __init__(self, block_index, vision_config, mm_type, ln_type, attn_type):
        super().__init__()
        self.block_index = block_index
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.act_fn_name = vision_config.hidden_act
        prefix = f"model.visual.blocks.{block_index}"
        self.add_module("norm1", LN_WEIGHT_REGISTER[ln_type](f"{prefix}.norm1.weight", f"{prefix}.norm1.bias", eps=1e-6))
        self.add_module("norm2", LN_WEIGHT_REGISTER[ln_type](f"{prefix}.norm2.weight", f"{prefix}.norm2.bias", eps=1e-6))
        self.add_module("qkv", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.attn.qkv.weight", f"{prefix}.attn.qkv.bias"))
        self.add_module("proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.attn.proj.weight", f"{prefix}.attn.proj.bias"))
        self.add_module("linear_fc1", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.mlp.linear_fc1.weight", f"{prefix}.mlp.linear_fc1.bias"))
        self.add_module("linear_fc2", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.mlp.linear_fc2.weight", f"{prefix}.mlp.linear_fc2.bias"))
        self.add_module("attn", ATTN_WEIGHT_REGISTER[attn_type]())


class HidreamO1ImageVisionPatchMergerWeights(WeightModule):
    def __init__(self, prefix, vision_config, mm_type, ln_type, use_postshuffle_norm):
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        self.hidden_size = vision_config.hidden_size * (vision_config.spatial_merge_size**2)
        self.add_module("norm", LN_WEIGHT_REGISTER[ln_type](f"{prefix}.norm.weight", f"{prefix}.norm.bias", eps=1e-6))
        self.add_module("linear_fc1", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.linear_fc1.weight", f"{prefix}.linear_fc1.bias"))
        self.add_module("linear_fc2", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.linear_fc2.weight", f"{prefix}.linear_fc2.bias"))
