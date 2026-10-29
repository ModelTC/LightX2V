import torch

from lightx2v.utils.envs import GET_DTYPE

from .flash_model_infer import FlashModelPostInfer, FlashModelPreInfer, FlashModelTransformerInfer
from .parallel import SequenceParallel
from .rope import RotaryEmbedding
from .wan_infer import RelayWanPostInfer, RelayWanPreInfer, RelayWanTransformerInfer
from .weights import FlashModelWeights, WanWeights, checkpoint_config, resolve_checkpoints


def validate_checkpoint_configs(wan, flash_model, geometry):
    expected_wan = {
        "patch_size": [1, 2, 2],
        "in_channels": 48,
        "out_channels": 48,
        "qk_norm": "rms_norm_across_heads",
        "cross_attn_norm": True,
        "num_frames_per_block": 1,
        "spatial_attention_mode": "local",
        "upscale_factor": geometry.scale,
    }
    expected_flash_model = {
        "variant": "S",
        "upscale_factor": geometry.scale,
        "window_size": 2,
        "condition_channels": 48,
        "condition_mode": "endpoints",
        "temporal_rope_policy": "current_relative",
        "dim": 512,
        "depth": 10,
        "num_heads": 8,
        "num_kv_groups": 2,
    }
    for actual, expected in ((wan, expected_wan), (flash_model, expected_flash_model)):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"Unsupported RelayVSR checkpoint: {key}={actual.get(key)!r}, expected {value!r}")
    if wan.get("temporal_rope", {}).get("policy") != "fixed_end":
        raise ValueError("RelayVSR requires fixed_end Wan temporal RoPE")


class RelayVSRModel:
    def __init__(self, config, geometry, group=None):
        self.config, self.geometry = config, geometry
        wan_path, flash_model_path = resolve_checkpoints(config, geometry)
        self.wan_config = checkpoint_config(wan_path, "wan", geometry)
        # "flash" is the component identifier in the official checkpoint format.
        self.flash_model_config = checkpoint_config(flash_model_path, "flash", geometry)
        validate_checkpoint_configs(self.wan_config, self.flash_model_config, geometry)
        # The distilled checkpoint relies on BF16/FP16 rounding between ops.
        # Keep those casts and avoid rewriting mm + residual into addmm.
        compile_config = {**config, "compile_options": {"emulate_precision_casts": True, "pattern_matcher": False, **config.get("compile_options", {})}}
        self.parallel = SequenceParallel(group)
        self.wan_weights = WanWeights(wan_path, self.wan_config, config)
        self.flash_model_weights = FlashModelWeights(flash_model_path, self.flash_model_config, geometry)
        self.wan_pre = RelayWanPreInfer(self.wan_config, geometry)
        self.wan_transformer = RelayWanTransformerInfer(self.wan_config, compile_config, self.parallel, self.wan_weights)
        self.wan_post = RelayWanPostInfer({"out_dim": self.wan_config["out_channels"]})
        self.flash_model_pre = FlashModelPreInfer()
        self.flash_model_transformer = FlashModelTransformerInfer(self.flash_model_config, self.parallel, compile_config)
        self.flash_model_post = FlashModelPostInfer(geometry)
        self.wan_rope = RotaryEmbedding(wan=True, device=torch.device("cuda"), dtype=GET_DTYPE(), head_dim=self.wan_config["attention_head_dim"], max_length=self.wan_config["rope_max_seq_len"])
        self.flash_model_rope = RotaryEmbedding(wan=False, device=torch.device("cuda"), dtype=GET_DTYPE(), theta=self.flash_model_config["rope_theta"])
        self.temb, self.modulation, self.context = self.wan_pre.conditioning(self.wan_weights.pre)
        self.wan_transformer.prepare_cross_attention(self.wan_weights.blocks, self.context)

    def set_scheduler(self, scheduler):
        self.wan_transformer.set_scheduler(scheduler)
        self.wan_post.set_scheduler(scheduler)

    def infer(self, scheduler, lr, cache, frame):
        height, width = self.geometry.wan_grid(*lr.shape[-2:])
        hidden = self.parallel.split(self.wan_pre.infer(self.wan_weights.pre, scheduler.latents.to(GET_DTYPE()), lr))
        pre = self.wan_transformer.prepare_input(hidden, self.temb, self.modulation, self.context, self.wan_rope, cache, frame, height, width)
        hidden = self.wan_transformer.infer(self.wan_weights, pre)
        scheduler.noise_pred = self.wan_post.infer(hidden, pre)

    def flash_model_prepare_condition(self, latent):
        height, width = latent.shape[-2:]
        hidden = self.parallel.split(self.flash_model_pre.condition(self.flash_model_weights.pre, latent))
        return self.flash_model_transformer.prepare_condition(self.flash_model_weights.blocks, hidden, self.flash_model_rope, height, width)

    def flash_model_decode(self, lr, conditions, history, frame):
        height, width = self.geometry.latent_grid(*lr.shape[-2:])
        hidden = self.parallel.split(self.flash_model_pre.frame(self.flash_model_weights.pre, lr))
        hidden = self.flash_model_transformer.infer(self.flash_model_weights.blocks, hidden, conditions, history, self.flash_model_rope, frame, height, width)
        return self.flash_model_post.infer(self.flash_model_weights.post, self.parallel.gather(hidden, height * width), lr)

    def clear(self):
        self.wan_rope.clear()
        self.flash_model_rope.clear()
