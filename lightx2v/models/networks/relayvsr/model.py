import torch

from lightx2v.utils.envs import GET_DTYPE

from .flash_infer import FlashPostInfer, FlashPreInfer, FlashTransformerInfer
from .parallel import SequenceParallel
from .rope import RotaryEmbedding
from .wan_infer import RelayWanPostInfer, RelayWanPreInfer, RelayWanTransformerInfer
from .weights import FlashWeights, WanWeights, checkpoint_config, resolve_checkpoints


def validate_checkpoint_configs(wan, flash):
    expected_wan = {
        "patch_size": [1, 2, 2],
        "in_channels": 48,
        "out_channels": 48,
        "qk_norm": "rms_norm_across_heads",
        "cross_attn_norm": True,
        "num_frames_per_block": 1,
        "spatial_attention_mode": "local",
    }
    expected_flash = {"variant": "S", "upscale_factor": 4, "window_size": 2, "condition_channels": 48, "condition_mode": "endpoints", "temporal_rope_policy": "current_relative"}
    for actual, expected in ((wan, expected_wan), (flash, expected_flash)):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"Unsupported RelayVSR checkpoint: {key}={actual.get(key)!r}, expected {value!r}")
    if wan.get("temporal_rope", {}).get("policy") != "fixed_end":
        raise ValueError("RelayVSR requires fixed_end Wan temporal RoPE")


class RelayVSRModel:
    def __init__(self, config, group=None):
        self.config = config
        wan_path, flash_path = resolve_checkpoints(config)
        self.wan_config = checkpoint_config(wan_path, "wan")
        self.flash_config = checkpoint_config(flash_path, "flash")
        validate_checkpoint_configs(self.wan_config, self.flash_config)
        # The distilled checkpoint relies on BF16/FP16 rounding between ops.
        # Keep those casts and avoid rewriting mm + residual into addmm.
        compile_config = {**config, "compile_options": {"emulate_precision_casts": True, "pattern_matcher": False, **config.get("compile_options", {})}}
        self.parallel = SequenceParallel(group)
        self.wan_weights = WanWeights(wan_path, self.wan_config, config)
        self.flash_weights = FlashWeights(flash_path, self.flash_config)
        self.wan_pre = RelayWanPreInfer(self.wan_config)
        self.wan_transformer = RelayWanTransformerInfer(self.wan_config, compile_config, self.parallel, self.wan_weights)
        self.wan_post = RelayWanPostInfer({"out_dim": self.wan_config["out_channels"]})
        self.flash_pre = FlashPreInfer()
        self.flash_transformer = FlashTransformerInfer(self.flash_config, self.parallel, compile_config)
        self.flash_post = FlashPostInfer()
        self.wan_rope = RotaryEmbedding(wan=True, device=torch.device("cuda"), dtype=GET_DTYPE(), head_dim=self.wan_config["attention_head_dim"], max_length=self.wan_config["rope_max_seq_len"])
        self.flash_rope = RotaryEmbedding(wan=False, device=torch.device("cuda"), dtype=GET_DTYPE(), theta=self.flash_config["rope_theta"])
        self.temb, self.modulation, self.context = self.wan_pre.conditioning(self.wan_weights.pre)
        self.wan_transformer.prepare_cross_attention(self.wan_weights.blocks, self.context)

    def set_scheduler(self, scheduler):
        self.wan_transformer.set_scheduler(scheduler)
        self.wan_post.set_scheduler(scheduler)

    def infer(self, scheduler, lr, cache, frame):
        height, width = lr.shape[-2] // 8, lr.shape[-1] // 8
        hidden = self.parallel.split(self.wan_pre.infer(self.wan_weights.pre, scheduler.latents.to(GET_DTYPE()), lr))
        pre = self.wan_transformer.prepare_input(hidden, self.temb, self.modulation, self.context, self.wan_rope, cache, frame, height, width)
        hidden = self.wan_transformer.infer(self.wan_weights, pre)
        scheduler.noise_pred = self.wan_post.infer(hidden, pre)

    def prepare_condition(self, latent):
        height, width = latent.shape[-2:]
        hidden = self.parallel.split(self.flash_pre.condition(self.flash_weights.pre, latent))
        return self.flash_transformer.prepare_condition(self.flash_weights.blocks, hidden, self.flash_rope, height, width)

    def decode(self, lr, conditions, history, frame):
        height, width = lr.shape[-2] // 4, lr.shape[-1] // 4
        hidden = self.parallel.split(self.flash_pre.frame(self.flash_weights.pre, lr))
        hidden = self.flash_transformer.infer(self.flash_weights.blocks, hidden, conditions, history, self.flash_rope, frame, height, width)
        return self.flash_post.infer(self.flash_weights.post, self.parallel.gather(hidden, height * width), lr)

    def clear(self):
        self.wan_rope.clear()
        self.flash_rope.clear()
