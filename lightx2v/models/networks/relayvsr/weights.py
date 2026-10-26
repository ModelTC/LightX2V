import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.norm.layer_norm_weight import LNWeight
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeight, RMSWeightNative
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, CONV2D_WEIGHT_REGISTER, CONV3D_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER

from .compile import native_layer_norm, native_rms_norm


class RelayRMSNorm(RMSWeightNative):
    def apply(self, x):
        if torch.compiler.is_compiling():
            return native_rms_norm(x, self._get_actual_weight(), self.eps)
        return super().apply(x)


class FP32LayerNorm(LNWeight):
    def __init__(self, *args, cast_output=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cast_output = cast_output

    def apply(self, x):
        if torch.compiler.is_compiling():
            output = native_layer_norm(x, self.weight, self.bias, self.eps)
        else:
            output = F.layer_norm(x.float(), (x.shape[-1],), None if self.weight is None else self.weight.float(), None if self.bias is None else self.bias.float(), self.eps)
        return output.to(x.dtype) if self.cast_output else output


class ChannelRMSNorm(RMSWeight):
    def apply(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + self.eps)
        return (value * self.weight).to(x.dtype)


class RelayWeights(WeightModule):
    def linear(self, name, key, *, bias=True, buffer=False):
        self.add_module(name, MM_WEIGHT_REGISTER["Default"](key + ".weight", key + ".bias" if bias else None, create_cuda_buffer=buffer))

    def tensor(self, name, key, *, buffer=False):
        self.add_module(name, TENSOR_REGISTER["Default"](key, create_cuda_buffer=buffer))

    def rms(self, name, key, *, eps, buffer=False):
        self.add_module(name, RelayRMSNorm(key + ".weight", eps=eps, create_cuda_buffer=buffer))

    def conv(self, name, key, *, stride=1, padding=0):
        self.add_module(name, CONV2D_WEIGHT_REGISTER["Default"](key + ".weight", key + ".bias", stride=stride, padding=padding))


class WanPreWeights(RelayWeights):
    def __init__(self, config):
        super().__init__()
        self.add_module("patch", CONV3D_WEIGHT_REGISTER["Default"]("patch_embedding.weight", "patch_embedding.bias", stride=tuple(config["patch_size"])))
        self.conv("lq_conv1", "lq_proj.conv1")
        self.conv("lq_conv2", "lq_proj.conv2", stride=2, padding=1)
        for i in (1, 2):
            self.add_module(f"lq_norm{i}", ChannelRMSNorm(f"lq_proj.norm{i}.weight", eps=1e-6))
        # Upstream's transposed tokens normally require separate matmul/bias
        # rounding. A one-token grid is contiguous and uses fused addmm instead.
        self.linear("lq_out", "lq_proj.out_proj")
        self.linear("lq_out_unbiased", "lq_proj.out_proj", bias=False)
        for name, key in (
            ("text1", "text_embedder.linear_1"),
            ("text2", "text_embedder.linear_2"),
            ("time1", "time_embedder.linear_1"),
            ("time2", "time_embedder.linear_2"),
            ("time_proj", "time_proj"),
        ):
            self.linear(name, "condition_embedder." + key)
        self.tensor("prompt", "prompt_embeds")


class WanBlockWeights(RelayWeights):
    """Official checkpoint names mapped onto the Wan family's three phases."""

    def __init__(self, index, config, *, attention_backend="torch_sdpa", buffer=False):
        super().__init__()
        prefix = f"blocks.{index}."
        phases = WeightModuleList([RelayWeights() for _ in range(3)])
        self.add_module("compute_phases", phases)
        for phase, attention, family_name in ((phases[0], "attn1", "self_attn"), (phases[1], "attn2", "cross_attn")):
            for projection in ("q", "k", "v"):
                phase.linear(family_name + "_" + projection, prefix + attention + ".to_" + projection, buffer=buffer)
            phase.linear(family_name + "_o", prefix + attention + ".to_out.0", buffer=buffer)
            for projection in ("q", "k"):
                phase.rms(family_name + "_norm_" + projection, prefix + attention + ".norm_" + projection, eps=config["eps"], buffer=buffer)
            phase.add_module(family_name + "_1", ATTN_WEIGHT_REGISTER["torch_sdpa" if attention == "attn1" else attention_backend]())
        phases[0].add_module("norm1", FP32LayerNorm(eps=config["eps"]))
        phases[0].tensor("modulation", prefix + "scale_shift_table", buffer=buffer)
        phases[1].add_module("norm3", FP32LayerNorm(prefix + "norm2.weight", prefix + "norm2.bias", eps=config["eps"], create_cuda_buffer=buffer, cast_output=True))
        phases[2].add_module("norm2", FP32LayerNorm(eps=config["eps"]))
        phases[2].linear("ffn_0", prefix + "ffn.net.0.proj", buffer=buffer)
        phases[2].linear("ffn_2", prefix + "ffn.net.2", buffer=buffer)


class WanPostWeights(RelayWeights):
    def __init__(self, config):
        super().__init__()
        self.add_module("norm", FP32LayerNorm(eps=config["eps"]))
        self.tensor("modulation", "scale_shift_table")
        self.linear("projection", "proj_out")


class FlashPreWeights(RelayWeights):
    def __init__(self, config):
        super().__init__()
        for i, stride, padding in ((0, 2, 1), (2, 2, 1), (4, 1, 1), (6, 1, 0)):
            self.conv(f"conv{i}", f"lr_encoder.layers.{i}", stride=stride, padding=padding)
        self.conv("condition", "condition_encoder.projection")
        self.rms("lr_norm", "lr_encoder.norm", eps=config["norm_eps"])
        self.rms("condition_norm", "condition_encoder.norm", eps=config["norm_eps"])
        self.tensor("video_type", "video_type_embedding")
        self.tensor("condition_type", "condition_type_embedding")


class FlashBlockWeights(RelayWeights):
    def __init__(self, index, config):
        super().__init__()
        prefix = f"backbone.blocks.{index}."
        for name in ("query", "key", "value", "condition_key", "condition_value", "output"):
            self.linear(name, prefix + "attention." + name)
        for name in ("key_norm", "value_norm"):
            self.rms(name, prefix + "attention." + name, eps=config["norm_eps"])
        for name in ("attention_norm", "feed_forward_norm"):
            self.rms(name, prefix + name, eps=config["norm_eps"])
        self.linear("gate_up", prefix + "feed_forward.gate_up_proj", bias=False)
        self.linear("down", prefix + "feed_forward.down", bias=False)


class FlashPostWeights(RelayWeights):
    def __init__(self):
        super().__init__()
        self.linear("head", "spatial_head.0")
        self.linear("projection", "residual_projection")


def checkpoint_config(path, component):
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != f"refvsr-{component}-v1":
        raise ValueError(f"Expected official RelayVSR {component} checkpoint with embedded config: {path}")
    return json.loads(metadata["config"])


def resolve_checkpoints(config):
    root = Path(config["model_path"])
    return Path(config.get("dit_original_ckpt") or root / "wan.safetensors"), Path(config.get("vae_original_ckpt") or root / "flash.safetensors")


def load_container(container, handle, keys, device):
    # Mapping and device placement happen once here, never in the inference path.
    state = {key: handle.get_tensor(key).to(device=device, dtype=GET_DTYPE()) for key in keys}
    container.load(state)


class WanWeights(WeightModule):
    def __init__(self, path, model_config, deployment):
        super().__init__()
        self.add_module("pre", WanPreWeights(model_config))
        self.add_module("post", WanPostWeights(model_config))
        attention_backend = deployment.get("cross_attention_backend", "torch_sdpa")
        blocks = WeightModuleList([WanBlockWeights(i, model_config, attention_backend=attention_backend) for i in range(model_config["num_layers"])])
        self.add_module("blocks", blocks)
        self.offload = deployment.get("cpu_offload", False)
        self.offload_block_cuda_buffers = []
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            non_blocks = [key for key in keys if not key.startswith("blocks.")]
            load_container(WeightModuleList([self.pre, self.post]), handle, non_blocks, "cuda")
            for i, block in enumerate(blocks):
                block_keys = [key for key in keys if key.startswith(f"blocks.{i}.")]
                load_container(block, handle, block_keys, "cpu" if self.offload else "cuda")
                if i == 0 and self.offload:
                    for _ in range(2):
                        slot = WanBlockWeights(0, model_config, attention_backend=attention_backend, buffer=True)
                        load_container(slot, handle, block_keys, "cuda")
                        self.offload_block_cuda_buffers.append(slot)


class FlashWeights(WeightModule):
    def __init__(self, path, config):
        super().__init__()
        self.add_module("pre", FlashPreWeights(config))
        self.add_module("blocks", WeightModuleList([FlashBlockWeights(i, config) for i in range(10)]))
        self.add_module("post", FlashPostWeights())
        with safe_open(path, framework="pt", device="cpu") as handle:
            # condition_rgb_projection is an auxiliary training head, unused by decode.
            keys = [key for key in handle.keys() if not key.startswith("condition_rgb_projection.")]  # noqa: SIM118 - safe_open is not iterable
            load_container(self, handle, keys, "cuda")
