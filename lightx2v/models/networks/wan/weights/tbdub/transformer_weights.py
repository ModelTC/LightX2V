import torch

from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights, _mm_weight, _rms_weight
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, ROPE_REGISTER


class TBDubTransformerWeights(WanTransformerWeights):
    """Wan weights plus frame-local audio attention in every cross-attention phase.

    Extend the real blocks and both offload buffer sets identically. The existing
    Wan block loader then transfers the audio parameters together with each block.
    """

    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__(config, lazy_load_path=lazy_load_path, lora_path=lora_path)
        block_sets = [self.blocks]
        for name in ("offload_block_cuda_buffers", "offload_block_cpu_buffers"):
            buffers = getattr(self, name, None)
            if buffers is not None:
                block_sets.append(buffers)
        for blocks in block_sets:
            for block in blocks:
                self_attn, cross_attn = block.compute_phases[:2]
                self_attn.add_module("rope", ROPE_REGISTER[config.get("rope_type", "torch_complex_rope")](layout="interleaved", compute_dtype=torch.float64))
                prefix = f"blocks.{block.block_index}"
                storage = dict(
                    create_cuda_buffer=block.create_cuda_buffer,
                    create_cpu_buffer=block.create_cpu_buffer,
                    lazy_load=block.lazy_load,
                    lazy_load_file=block.lazy_load_file,
                    lora_prefix="blocks",
                    lora_path=lora_path,
                )
                cross_attn.add_module("norm4", LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](f"{prefix}.norm4.weight", f"{prefix}.norm4.bias", **storage))
                for name in ("q", "k", "v", "o"):
                    cross_attn.add_module(f"audio_{name}", _mm_weight(config, f"{prefix}.audio_attn.{name}.weight", f"{prefix}.audio_attn.{name}.bias", **storage))
                for name in ("q", "k"):
                    cross_attn.add_module(f"audio_norm_{name}", _rms_weight(config, f"{prefix}.audio_attn.norm_{name}.weight", **storage))
                cross_attn.add_module("audio_attn", ATTN_WEIGHT_REGISTER[config["cross_attn_1_type"]]())
