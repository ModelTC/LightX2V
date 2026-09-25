"""VDN-H3 branch weights, local linear attention and startup adapter merging.

Linear branch adapted from OpenVDN/vdn-minimax-h3 e02ff077 (Apache-2.0):
models/linear_attention/{branch,delta_rule,layers,features}.py.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from lightx2v.common.ops.attn.kernels.vdn import activate, linear_epilogue, temporal_activate
from lightx2v.common.ops.attn.linear_attn import LinearAttentionBase
from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight
from lightx2v.common.ops.tensor.tensor import DefaultTensor
from lightx2v_platform.base.global_var import AI_DEVICE


def configure_vdn(config):
    checkpoint = Path(config["vdn_checkpoint"]).expanduser().resolve()
    with (checkpoint / "model_spec.json").open() as handle:
        spec = json.load(handle)
    expected_attention = {
        "anchor_frames": "both",
        "enable_softmax_gate": True,
        "linear_attention": {
            "a_fp32": True,
            "bridge": "alpha",
            "delta_rule": "vdn_solve",
            "enable_text_state": True,
            "linear_head_dim": 128,
            "short_conv": {"targets": ["k", "v"]},
        },
        "softmax_attention": {"chunk": 5, "radius": 1},
    }
    expected_transform = {"type": "hybrid_attention", "version": 2, "config": expected_attention}
    if spec.get("format_version") != 2 or spec.get("transforms") != [expected_transform]:
        raise ValueError("VDN requires the released v2 alpha/vdn_solve hybrid attention with both anchors, text state, head dim 128 and K/V short convolutions")
    if spec.get("base", {}).get("subfolder") != "transformer" or config.get("model_variant") != "fl2av":
        raise ValueError("VDN-H3 requires model_variant='fl2av' and the base transformer checkpoint")
    if int(config.get("attention_head_dim", 128)) != 128 or tuple(config.get("patch_size", (1, 2, 2))) != (1, 2, 2):
        raise ValueError("VDN-H3 requires attention_head_dim=128 and patch_size=[1, 2, 2]")
    if AI_DEVICE != "cuda":
        raise NotImplementedError("VDN-H3 currently supports NVIDIA CUDA devices")
    unsupported = [key for key in ("dit_quantized", "weight_auto_quant", "shared_cpu_weights", "lora_dynamic_apply", "use_compile", "dummy_model", "lazy_load") if config.get(key, False)]
    if unsupported or config.get("lora_configs"):
        raise ValueError(f"VDN-H3 requires unquantized startup-merged weights without external LoRA; unsupported settings: {unsupported or ['lora_configs']}")
    communication_quant = (config.get("parallel") or {}).get("seq_p_quant_scheme")
    if communication_quant is not None and communication_quant is not False:
        raise ValueError("VDN-H3 does not support quantized sequence-parallel communication")
    adapters = spec.get("adapters", [])
    if [adapter.get("config", {}).get("name", "default") for adapter in adapters] != ["default", "turbo"]:
        raise ValueError("VDN-H3 requires the ordered default and turbo adapters")
    for adapter in adapters:
        values = adapter["config"]
        if adapter.get("type") != "lora" or adapter.get("version") != 1 or values.get("rank", 0) <= 0 or values.get("alpha") != values.get("rank"):
            raise ValueError("VDN-H3 requires v1 LoRA adapters with alpha/rank=1")
        ranks, alphas = values.get("rank_pattern", {}), values.get("alpha_pattern", {})
        if ranks != alphas or any(rank <= 0 for rank in ranks.values()):
            raise ValueError("VDN-H3 requires alpha/rank=1 for every adapter target")
    config["vdn_checkpoint"] = str(checkpoint)
    config["vdn_attention"] = expected_attention


def vdn_adapter_paths(config):
    checkpoint = Path(config["vdn_checkpoint"])
    return [(name, checkpoint / "adapters" / name / "adapter_model.safetensors") for name in ("default", "turbo")]


def open_vdn_adapters(config, stack):
    """Open and validate adapter metadata once for a complete loading pass."""
    adapters = []
    target_shapes = {}
    for name, path in vdn_adapter_paths(config):
        source = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
        suffix = f".lora_A.{name}.weight"
        keys = set(source.keys())
        pairs = {}
        for a_key in sorted(key for key in keys if key.endswith(suffix)):
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            if b_key not in keys:
                raise ValueError(f"Missing VDN LoRA tensor: {b_key}")
            a_shape, b_shape = source.get_slice(a_key).get_shape(), source.get_slice(b_key).get_shape()
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
                raise ValueError(f"Invalid VDN LoRA shapes for {a_key}")
            target = a_key[: -len(suffix)].replace(".attn.orig.", ".attn.") + ".weight"
            shape = (b_shape[0], a_shape[1])
            if target in pairs or target_shapes.get(target, shape) != shape:
                raise ValueError(f"Conflicting VDN LoRA target: {target}")
            pairs[target] = (a_key, b_key)
            target_shapes[target] = shape
        if not pairs or keys != {key for pair in pairs.values() for key in pair}:
            raise ValueError(f"Incomplete or unsupported VDN {name} adapter tensors in {path}")
        adapters.append((source, pairs))
    checkpoint = Path(config["dit_original_ckpt"])
    files = sorted(checkpoint.glob("*.safetensors")) if checkpoint.is_dir() else [checkpoint]
    remaining = set(target_shapes)
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in remaining.intersection(source.keys()):
                if tuple(source.get_slice(key).get_shape()) != target_shapes[key]:
                    raise ValueError(f"VDN LoRA shape does not match base tensor: {key}")
                remaining.remove(key)
    if remaining:
        raise ValueError(f"VDN LoRA targets missing from the base checkpoint: {sorted(remaining)[:4]}")
    return adapters


@torch.no_grad()
def merge_vdn_tensor(tensor, key, adapters):
    """Merge on CPU in spec order: FP32 B@A, cast delta, add in base dtype."""
    for source, pairs in adapters:
        pair = pairs.get(key)
        if pair is None:
            continue
        a, b = (source.get_tensor(name).float() for name in pair)
        if tuple(tensor.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"VDN LoRA shape mismatch for {key}")
        if tensor.device.type != "cpu":
            raise ValueError("VDN adapters must be merged on CPU before TP sharding")
        tensor.add_((b @ a).to(tensor.dtype))
    return tensor


def validate_vdn_branch(source, config):
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    dim = int(config["vdn_attention"]["linear_attention"]["linear_head_dim"])
    channels = heads * dim
    shapes = {
        "linear_attention.alpha.A_log": (heads,),
        "linear_attention.alpha.down.weight": (dim, hidden),
        "linear_attention.alpha.dt_bias": (channels,),
        "linear_attention.alpha.up.weight": (channels, dim),
        "linear_attention.beta_proj.weight": (heads, hidden),
        "linear_attention.norm.weight": (dim,),
        "linear_attention.output_gate.down.weight": (dim, hidden),
        "linear_attention.output_gate.up.bias": (channels,),
        "linear_attention.output_gate.up.weight": (channels, dim),
        "linear_attention.short_conv.k_sp.weight": (channels, 1, 5, 5),
        "linear_attention.short_conv.k_tm.weight": (channels, 1, 5),
        "linear_attention.short_conv.v_sp.weight": (channels, 1, 5, 5),
        "linear_attention.short_conv.v_tm.weight": (channels, 1, 5),
        "softmax_gate.up.bias": (heads,),
        "softmax_gate.up.weight": (heads, hidden),
        "to_out_linear.weight": (hidden, channels),
    }
    expected = {f"transformer_blocks.{index}.attn.{suffix}": shape for index in range(int(config["num_layers"])) for suffix, shape in shapes.items()}
    if set(source.keys()) != expected.keys():
        raise ValueError("VDN linear branch tensors do not match the configured H3 blocks")
    for key, shape in expected.items():
        tensor = source.get_slice(key)
        if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != "BF16":
            raise ValueError(f"VDN branch {key} must be BF16 with shape {shape}")


class _VDNTensor(DefaultTensor):
    def __init__(self, tensor_name, create_cuda_buffer=False):
        super().__init__(tensor_name, create_cuda_buffer=create_cuda_buffer)
        self.base_attrs = [(tensor_name, "tensor", False)]

    @property
    def weight(self):
        return self.tensor


class MiniMaxH3VDNWeights(LinearAttentionBase):
    """Hybrid branch weights and linear computation, using native offload storage."""

    def __init__(self, prefix, config, linear_factory, create_cuda_buffer=False):
        super().__init__()
        linears = {
            "alpha_down": ("linear_attention.alpha.down", False, None),
            "alpha_up": ("linear_attention.alpha.up", False, "col"),
            "beta_proj": ("linear_attention.beta_proj", False, "col"),
            "output_gate_down": ("linear_attention.output_gate.down", False, None),
            "output_gate_up": ("linear_attention.output_gate.up", True, "col"),
            "softmax_gate": ("softmax_gate.up", True, "col"),
            "to_out_linear": ("to_out_linear", False, "row"),
        }
        for name, (suffix, bias, tp_split) in linears.items():
            self.add_module(name, linear_factory(config, f"{prefix}.{suffix}", bias=bias, create_cuda_buffer=create_cuda_buffer, tp_split=tp_split))
        tensors = {
            "alpha_a_log": "linear_attention.alpha.A_log",
            "alpha_dt_bias": "linear_attention.alpha.dt_bias",
            "norm": "linear_attention.norm.weight",
            "k_sp": "linear_attention.short_conv.k_sp.weight",
            "k_tm": "linear_attention.short_conv.k_tm.weight",
            "v_sp": "linear_attention.short_conv.v_sp.weight",
            "v_tm": "linear_attention.short_conv.v_tm.weight",
        }
        for name, suffix in tensors.items():
            self.add_module(name, _VDNTensor(f"{prefix}.{suffix}", create_cuda_buffer=create_cuda_buffer))

    @staticmethod
    def _factor_states(alpha, a, b):
        with torch.autocast(device_type=a.device.type, enabled=False):
            eye = torch.eye(a.shape[-1], device=a.device, dtype=torch.float32).expand_as(a)
            chol = torch.linalg.cholesky(a.float() + eye)
            inverse_l = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
            inverse = inverse_l.transpose(-1, -2) @ inverse_l
            return alpha.unsqueeze(-1) * inverse, b.float() @ inverse

    @staticmethod
    def _gather_states(prefix, suffix, alpha, text_state, bounds):
        count = alpha.shape[0]
        device = alpha.device
        before = torch.tensor([lo - 1 for lo, _ in bounds], device=device)
        after = torch.tensor([hi + 1 for _, hi in bounds], device=device)
        left = torch.where((before >= 0)[:, None, None, None], prefix[before.clamp(min=0)], text_state)
        right = torch.where((after < count)[:, None, None, None], suffix[after.clamp(max=count - 1)], text_state)
        cumulative = torch.cat((torch.zeros_like(alpha[:1]), torch.log(alpha.clamp_min(1e-12)).cumsum(0)))
        frame = torch.arange(count, device=device)
        left_decay = torch.exp(cumulative[frame + 1] - cumulative[(before + 1).clamp(min=0)])
        right_decay = torch.exp(cumulative[after.clamp(max=count)] - cumulative[frame])
        return left * left_decay.unsqueeze(2) + right * right_decay.unsqueeze(2)

    @staticmethod
    def _conv_feature(tokens, spatial_weight, temporal_weight, frames, frame_size, normalize):
        heads, dim = tokens.shape[-2:]
        height, width = frame_size
        channels = heads * dim
        volume = tokens.reshape(frames, height, width, channels).permute(0, 3, 1, 2)
        volume = F.conv2d(volume, spatial_weight, padding=2, groups=channels)
        x = volume.permute(0, 2, 3, 1).reshape(frames, height * width, channels)
        temporal = temporal_weight.squeeze(1).to(x.dtype).contiguous()
        return temporal_activate(x, temporal, heads, dim, normalize)

    def frame_alpha(self, means, head_start, heads, dim):
        channels = slice(head_start * dim, (head_start + heads) * dim)
        with torch.autocast(device_type=means.device.type, enabled=False):
            # MMWeight stores transposed [in, out] weights. Alpha uses FP32 even
            # though the learned projections are stored in BF16.
            delta = means.float() @ self.alpha_down.weight.float()
            delta = delta @ unwrap_tp_weight(self.alpha_up).weight[:, channels].float()
            delta = delta + self.alpha_dt_bias.weight[channels].float()
            scale = self.alpha_a_log.weight[head_start : head_start + heads].float().exp()[:, None]
            return torch.exp(-scale * F.softplus(delta.view(-1, heads, dim)))

    def apply(self, q, k, v, *, alpha, beta, gate, layout, head_start=0, use_tf32=True):
        """Compute linear attention on a full sequence and a local head shard.

        Q/K/V are shared projections before softmax QK norm and RoPE. head_start
        selects this SP shard's heads within the TP-local convolution weights.
        Communication and the output projection stay in the caller. Non-video
        and generated boundary rows remain zero.
        """
        heads, dim = q.shape[-2:]
        frames, spatial = layout.num_frames, layout.tokens_per_frame
        output = q.new_zeros(q.shape)
        if frames <= 2 or layout.full_cover:
            return output
        channels = slice(head_start * dim, (head_start + heads) * dim)
        text_length = layout.text_length
        # The Qwen prefix includes vision rows. Text uses no spatial/temporal conv.
        text_k = F.normalize(F.silu(k[:text_length]), dim=-1, eps=1e-6).to(k.dtype)
        text_v = F.silu(v[:text_length])
        a_text, b_text = self.weighted_statistics(text_k.permute(1, 0, 2)[None], text_v.permute(1, 0, 2)[None], beta[:text_length].transpose(0, 1)[None])
        _, injection = self._factor_states(torch.ones((1, heads, dim), device=q.device), a_text, b_text)
        text_state = injection[0] * 0.5
        del text_k, text_v, a_text, b_text, injection

        # Both generated boundary latent frames belong entirely to dense softmax.
        start = layout.video_start + spatial
        stop = layout.video_start + (frames - 1) * spatial
        count = frames - 2
        # Match the contiguous SP1 reduction layout after Ulysses packs Q/K/V.
        q = activate(q[start:stop].contiguous(), True).view(count, spatial, heads, dim).permute(0, 2, 1, 3).contiguous()
        k = self._conv_feature(k[start:stop], self.k_sp.weight[channels], self.k_tm.weight[channels], count, layout.frame_size, True)
        v = self._conv_feature(v[start:stop], self.v_sp.weight[channels], self.v_tm.weight[channels], count, layout.frame_size, False)
        k = k.view(count, spatial, heads, dim).permute(0, 2, 1, 3)
        v = v.view(count, spatial, heads, dim).permute(0, 2, 1, 3)
        frame_beta = beta[start:stop].view(count, spatial, heads).permute(0, 2, 1)
        # The released VDN inference path uses TF32 only for this video A statistic.
        a, b = self.weighted_statistics(k, v, frame_beta, use_tf32=use_tf32)
        del k, v, frame_beta
        transitions, injections = self._factor_states(alpha, a, b)
        prefix = self.affine_scan(transitions, injections, text_state)
        suffix = self.affine_scan(transitions, injections, text_state, reverse=True)
        del transitions, injections
        bounds = [(lo - 1, hi - 1) for lo, hi in layout.bounds[1:-1]]
        state = self._gather_states(prefix, suffix, alpha, text_state, bounds).to(gate.dtype)
        del prefix, suffix, a, b
        readout = q @ state.transpose(-1, -2)
        output[start:stop] = linear_epilogue(readout, self.norm.weight, gate[start:stop])
        return output
