"""VDN-H3 hybrid attention weights, computation and startup adapter merging.

Hybrid attention adapted from OpenVDN/vdn-minimax-h3 e02ff077 (Apache-2.0):
models/linear_attention/{branch,delta_rule,layers,features,kernels}.py and
models/ops/temporal_conv.py.
"""

import json
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from lightx2v.common.ops.attn.linear_attn import LinearAttentionBase
from lightx2v.common.ops.attn.template import AttnWeightTemplate
from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight
from lightx2v.common.ops.tensor.tensor import DefaultTensor
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


@cache
def _compiled(fn):
    return torch.compile(fn, dynamic=False)


def _activate_body(x, normalize):
    y = F.silu(x)
    return F.normalize(y, dim=-1, eps=1e-6).to(y.dtype) if normalize else y


def _activate(x, normalize):
    return _compiled(_activate_body)(x, normalize) if x.is_cuda else _activate_body(x, normalize)


def _epilogue_body(readout, weight, gate):
    ms = torch.linalg.vector_norm(readout, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / readout.shape[-1]
    normed = readout * torch.rsqrt(ms + 1e-6).to(readout.dtype) * weight.to(readout.dtype)
    frames, heads, spatial, dim = normed.shape
    return normed.permute(0, 2, 1, 3).reshape(frames * spatial, heads, dim) * gate


def _linear_epilogue(readout, weight, gate):
    return _compiled(_epilogue_body)(readout, weight, gate) if readout.is_cuda else _epilogue_body(readout, weight, gate)


if triton is not None:

    @triton.jit
    def _temporal_kernel(X, W, OUT, T, S, C, D: tl.constexpr, L2: tl.constexpr, BLOCK_T: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        channels = tl.program_id(2) * D + tl.arange(0, D)
        spatial = tl.program_id(1)
        acc = tl.zeros((BLOCK_T, D), tl.float32)
        for tap in tl.static_range(5):
            frames = rows + tap - 2
            x = tl.load(X + (frames[:, None] * S + spatial) * C + channels[None], mask=(rows[:, None] < T) & (frames[:, None] >= 0) & (frames[:, None] < T), other=0).to(tl.float32)
            w = tl.load(W + channels * 5 + tap).to(tl.float32)
            acc += x * w[None]
        y = acc * tl.sigmoid(acc)
        if L2:
            y *= tl.rsqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))[:, None]
        tl.store(OUT + (rows[:, None] * S + spatial) * C + channels[None], y, mask=rows[:, None] < T)


def _temporal_activate(x, weight, heads, dim, normalize):
    """Symmetric five-tap depthwise temporal conv, SiLU and optional L2 norm."""
    frames, spatial, channels = x.shape
    if x.is_cuda:
        if triton is None:
            raise ImportError("VDN CUDA inference requires Triton")
        if not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("VDN temporal convolution requires contiguous inputs")
        out = torch.empty_like(x)
        _temporal_kernel[(triton.cdiv(frames, 16), spatial, heads)](x, weight, out, frames, spatial, channels, D=dim, L2=normalize, BLOCK_T=16, num_warps=4, num_stages=2)
        return out.reshape(-1, heads, dim)
    padded = F.pad(x.float(), (0, 0, 0, 0, 2, 2))
    out = sum(padded[tap : tap + frames] * weight[:, tap].float() for tap in range(5))
    return _activate_body(out.reshape(-1, heads, dim), normalize).to(x.dtype)


@dataclass(frozen=True)
class VDNLayout:
    sequence_length: int
    video_start: int
    num_frames: int
    frame_size: tuple[int, int]
    text_length: int
    chunk: int = 5
    radius: int = 1

    @property
    def tokens_per_frame(self):
        return self.frame_size[0] * self.frame_size[1]

    @property
    def video_end(self):
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def bounds(self):
        return tuple(((frame // self.chunk - self.radius) * self.chunk, (frame // self.chunk + self.radius + 1) * self.chunk - 1) for frame in range(self.num_frames))

    @property
    def full_cover(self):
        return all(lo <= 0 and hi >= self.num_frames - 1 for lo, hi in self.bounds)


def _build_window_block_mask(layout, device, block_size=128):
    """Build block sparsity directly from frame windows."""
    start, end, spatial, frames = layout.video_start, layout.video_end, layout.tokens_per_frame, layout.num_frames
    chunk, radius = layout.chunk, layout.radius

    def mask_mod(batch, head, query, key):
        query_video = (query >= start) & (query < end)
        key_video = (key >= start) & (key < end)
        q_frame = ((query - start) // spatial).clamp(0, frames - 1)
        k_frame = (key - start) // spatial
        lo, hi = (q_frame // chunk - radius) * chunk, (q_frame // chunk + radius + 1) * chunk - 1
        window = (k_frame >= lo) & (k_frame <= hi)
        anchors = (q_frame == 0) | (q_frame == frames - 1) | (k_frame == 0) | (k_frame == frames - 1)
        return (~(query_video & key_video)) | window | anchors

    count = (layout.sequence_length + block_size - 1) // block_size
    columns = torch.arange(count)
    k_start = columns * block_size
    k_end = (k_start + block_size).clamp(max=layout.sequence_length)
    partial_indices = torch.zeros((1, 1, count, count), dtype=torch.int32)
    full_indices = torch.zeros_like(partial_indices)
    partial_count = torch.zeros((1, 1, count), dtype=torch.int32)
    full_count = torch.zeros_like(partial_count)
    # Global tokens/anchors are dense. Other rows use the union/intersection
    # of their frame windows for partial/full blocks.
    normal_start, normal_end = start + spatial, end - spatial
    for row in range(count):
        q_start, q_end = row * block_size, min((row + 1) * block_size, layout.sequence_length)
        first, last = max(q_start, normal_start), min(q_end, normal_end)
        if first >= last:
            any_allowed = torch.ones(count, dtype=torch.bool)
            all_allowed = any_allowed.clone()
        else:
            first_frame, last_frame = (first - start) // spatial, (last - 1 - start) // spatial
            union_lo = start + (first_frame // chunk - radius) * chunk * spatial
            union_hi = start + (last_frame // chunk + radius + 1) * chunk * spatial
            intersect_lo = max(normal_start, min(normal_end, start + (last_frame // chunk - radius) * chunk * spatial))
            intersect_hi = max(normal_start, min(normal_end, start + (first_frame // chunk + radius + 1) * chunk * spatial))
            any_allowed = (k_start < normal_start) | (k_end > normal_end) | ((k_start < union_hi) & (k_end > union_lo))
            if q_start < first or q_end > last:
                any_allowed.fill_(True)
            left_gap = (intersect_lo > normal_start) & (k_start < intersect_lo) & (k_end > normal_start)
            right_gap = (intersect_hi < normal_end) & (k_start < normal_end) & (k_end > intersect_hi)
            all_allowed = ~(left_gap | right_gap)
        # Match the token mask's zero padding in the last block.
        all_allowed &= (k_end - k_start == block_size) & (q_end - q_start == block_size)
        partial = columns[any_allowed & ~all_allowed]
        full = columns[all_allowed]
        partial_count[0, 0, row] = partial.numel()
        full_count[0, 0, row] = full.numel()
        partial_indices[0, 0, row, : partial.numel()] = partial
        full_indices[0, 0, row, : full.numel()] = full
    stats = {
        "block_size": block_size,
        "grid_blocks": count * count,
        "partial_blocks": int(partial_count.sum()),
        "full_blocks": int(full_count.sum()),
        "index_bytes": sum(t.numel() * t.element_size() for t in (partial_count, partial_indices, full_count, full_indices)),
    }
    stats["active_block_fraction"] = (stats["partial_blocks"] + stats["full_blocks"]) / stats["grid_blocks"]
    mask = BlockMask.from_kv_blocks(
        partial_count,
        partial_indices,
        full_count,
        full_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(layout.sequence_length, layout.sequence_length),
        compute_q_blocks=False,
    )
    return mask.to(device), stats


@lru_cache(maxsize=2)
def _shared_window_mask(layout, device):
    return _build_window_block_mask(layout, torch.device(device))


@cache
def _compiled_window():
    return torch.compile(flex_attention, dynamic=False, fullgraph=True)


class VDNWindowAttention(AttnWeightTemplate):
    def __init__(self):
        self.config = {}
        self.backend = "flex_triton"
        self.layout = None
        self.mask = None
        self._mask_key = None
        self.block_stats = None

    def prepare(self, layout, device):
        self.layout = layout
        device = torch.device(device)
        if device.type != "cuda":
            self.backend = "sdpa_reference"
            return
        self.backend = "flex_triton"
        key = (layout, str(device))
        if key == self._mask_key:
            return
        self.mask, self.block_stats = _shared_window_mask(*key)
        self._mask_key = key

    def apply(self, q, k, v, **kwargs):
        if q.shape[0] != self.layout.sequence_length or k.shape[0] != self.layout.sequence_length:
            raise ValueError("VDN window attention needs the complete packed sequence on each head rank")
        if q.is_cuda:
            out = _compiled_window()(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=self.mask, scale=q.shape[-1] ** -0.5, kernel_options={"BACKEND": "TRITON"})
            return out[0].transpose(0, 1)
        return self._reference(q, k, v)

    def _reference(self, q, k, v):
        layout = self.layout
        output = torch.empty_like(q)
        global_rows = torch.cat((torch.arange(layout.video_start, device=q.device), torch.arange(layout.video_end, layout.sequence_length, device=q.device)))

        def attend(query, rows):
            return F.scaled_dot_product_attention(query.transpose(0, 1)[None], k[rows].transpose(0, 1)[None], v[rows].transpose(0, 1)[None])[0].transpose(0, 1)

        if global_rows.numel():
            output[global_rows] = attend(q[global_rows], torch.arange(layout.sequence_length, device=q.device))
        for frame, (lo, hi) in enumerate(layout.bounds):
            if frame in (0, layout.num_frames - 1):
                lo, hi = 0, layout.num_frames - 1
            lo, hi = max(0, lo), min(layout.num_frames - 1, hi)
            extras = [f for f in (0, layout.num_frames - 1) if not lo <= f <= hi]
            rows = [global_rows, torch.arange(layout.video_start + lo * layout.tokens_per_frame, layout.video_start + (hi + 1) * layout.tokens_per_frame, device=q.device)]
            rows.extend(torch.arange(layout.video_start + f * layout.tokens_per_frame, layout.video_start + (f + 1) * layout.tokens_per_frame, device=q.device) for f in extras)
            start = layout.video_start + frame * layout.tokens_per_frame
            output[start : start + layout.tokens_per_frame] = attend(q[start : start + layout.tokens_per_frame], torch.cat(rows))
        return output


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
        return _temporal_activate(x, temporal, heads, dim, normalize)

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
        q = _activate(q[start:stop].contiguous(), True).view(count, spatial, heads, dim).permute(0, 2, 1, 3).contiguous()
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
        output[start:stop] = _linear_epilogue(readout, self.norm.weight, gate[start:stop])
        return output
