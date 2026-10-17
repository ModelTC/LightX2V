import math

import torch
import torch.nn.functional as F

from lightx2v.common.ops.norm.rms_norm_weight import RMSWeight
from lightx2v.common.ops.norm.torch_compatible_rms_norm import can_use_torch_rms_norm_5120, torch_rms_norm_5120
from lightx2v.common.ops.rope import FlashInferRope, RopeTemplate, TorchRealRope
from lightx2v.models.networks.wan.infer.dreamzero.fused_ops import can_use_seg_kernels, can_use_torch_layer_norm_5120, fused_layer_norm, seg_gate_residual_, seg_modulate
from lightx2v.models.networks.wan.infer.dreamzero.pre_infer import _category_linear
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.infer.triton_ops import apply_rotary_embedding, fuse_scale_shift_kernel
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, ROPE_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


def _can_use_cuda_kernels(tensor):
    return AI_DEVICE == "cuda" and tensor.device.type == AI_DEVICE


@ROPE_REGISTER("dreamzero_rope")
class DreamZeroRope(RopeTemplate):
    def __init__(self, layout="interleaved", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "interleaved":
            raise ValueError("DreamZeroRope only supports interleaved layout.")
        self.torch_rope = TorchRealRope(layout=layout, compute_dtype=compute_dtype)
        self.flashinfer_rope = FlashInferRope(layout=layout, compute_dtype=compute_dtype)

    def prepare_freqs(self, freqs, rotary_dim=None):
        if torch.is_tensor(freqs) and torch.is_complex(freqs):
            freqs = torch.cat(
                [freqs.real.reshape(freqs.shape[0], -1), freqs.imag.reshape(freqs.shape[0], -1)],
                dim=-1,
            )
        elif isinstance(freqs, tuple):
            freqs = self.flashinfer_rope.prepare_freqs(freqs, rotary_dim=rotary_dim)
        elif not torch.is_tensor(freqs):
            raise TypeError(f"Unsupported DreamZero RoPE frequency type: {type(freqs)!r}")
        if rotary_dim is not None and freqs.shape[-1] != rotary_dim:
            raise ValueError(f"DreamZero RoPE frequency width must be {rotary_dim}, got {freqs.shape[-1]}.")
        return freqs.to(dtype=self.compute_dtype).contiguous()

    def prepare_positions(self, freqs):
        return torch.arange(freqs.shape[0], device=freqs.device, dtype=torch.long)

    def apply(self, q, k, freqs, positions=None, **kwargs):
        freqs = self.prepare_freqs(freqs, rotary_dim=q.shape[-1])
        if freqs.shape[0] != q.shape[0]:
            raise ValueError(f"DreamZero RoPE length mismatch: freqs={freqs.shape[0]}, q={q.shape[0]}.")
        if positions is not None and positions.shape[0] != q.shape[0]:
            raise ValueError(f"DreamZero RoPE position length mismatch: positions={positions.shape[0]}, q={q.shape[0]}.")
        if _can_use_cuda_kernels(q) and self.flashinfer_rope.is_available():
            return self.flashinfer_rope.apply(q, k, freqs, positions=positions)
        if _can_use_cuda_kernels(q):
            cos, sin = freqs.chunk(2, dim=-1)
            return (
                apply_rotary_embedding(q.contiguous(), cos, sin, interleaved=False),
                apply_rotary_embedding(k.contiguous(), cos, sin, interleaved=False),
            )
        return self.torch_rope.apply(q, k, freqs.to(q.device))

    def apply_single(self, x, freqs, **kwargs):
        output, _ = self.apply(x, x.clone(), freqs, **kwargs)
        return output


class _SegModulation:
    """A modulation tensor stored as unique rows ``rows[seg_idx[token]]``."""

    __slots__ = ("rows", "seg_idx")

    def __init__(self, rows, seg_idx):
        self.rows = rows  # [num_segments, dim] (may be a strided view)
        self.seg_idx = seg_idx  # int32 [tokens]

    def tokens(self):
        """Materialise the per-token [tokens, dim] tensor (fallback paths)."""
        return self.rows.index_select(0, self.seg_idx)


class DreamZeroTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.device = AI_DEVICE
        if AI_DEVICE == "cuda" and torch.cuda.is_available():
            self.device = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
        # The Thor kernels and the fused Triton paths below were verified bitwise
        # only on Thor; other devices keep the original ops.
        self.is_thor = "thor" in self.device.split()
        self.fuse_rms_norm = self.is_thor and config.get("dreamzero_fuse_rms_norm", True)
        # Residual + LayerNorm + modulation fused into one Triton pass (bitwise
        # identical to the unfused torch ops on the supported torch/shape).
        self.fuse_layer_norm = self.is_thor and config.get("dreamzero_fuse_layer_norm", True)
        self.num_action_per_block = int(config.get("num_action_per_block", config.get("action_horizon", 24)))
        self.num_state_per_block = int(config.get("num_state_per_block", 1))
        self.frame_seqlen = int(config["frame_seqlen"])
        local_attn_size = config.get("local_attn_size")
        if local_attn_size is None:
            max_chunk_size = int(config.get("max_chunk_size", 4))
            local_attn_size = max_chunk_size * int(config.get("num_frame_per_block", 2)) + 1
        self.max_attention_size = int(local_attn_size) * self.frame_seqlen
        # Tokens written by one forward: video frames of a block + action + state.
        self.max_new_tokens = int(config.get("num_frame_per_block", 2)) * self.frame_seqlen + self.num_action_per_block + self.num_state_per_block
        self.kv_caches = {}
        self.cross_attn_kv_caches = {}
        self._cu_seqlens_cache = {}

    def create_empty_kv_cache(self, dtype, device):
        """Per-block K/V buffers laid out so attention reads a contiguous view.

        Rows [0, lengths[b]) hold the cached (post-RoPE) video K/V.  Each forward
        writes its current K/V (video tokens followed by action/state tokens)
        directly at rows [lengths[b], lengths[b] + seq), so the attention K/V is
        the view ending there; nothing is copied into a scratch buffer.  The
        buffers are allocated lazily with room for ``capacity`` cached rows plus
        one forward's tokens.
        """
        capacity = max(int(self.max_attention_size), 1)
        return {
            "capacity": capacity,
            "window_size": int(self.max_attention_size),
            "dtype": dtype,
            "device": device,
            "k": [None for _ in range(self.blocks_num)],
            "v": [None for _ in range(self.blocks_num)],
            "lengths": [0 for _ in range(self.blocks_num)],
        }

    def get_kv_cache(self, cache_name, dtype, device):
        cache = self.kv_caches.get(cache_name)
        if cache is None or cache["dtype"] != dtype or cache["device"] != device:
            cache = self.create_empty_kv_cache(dtype=dtype, device=device)
            self.kv_caches[cache_name] = cache
        return cache

    def set_kv_cache(self, cache_name, cache):
        self.kv_caches[cache_name] = cache

    def _kv_buffers(self, cache, block_idx, needed):
        """Return this block's K/V buffers with at least ``needed`` rows."""
        k_buf = cache["k"][block_idx]
        if k_buf is not None and k_buf.shape[0] >= needed:
            return k_buf, cache["v"][block_idx]
        size = needed
        if int(cache["window_size"]) > 0:
            size = max(size, cache["capacity"] + self.max_new_tokens)
        if k_buf is not None:
            size = max(size, 2 * k_buf.shape[0])
        new_k = torch.empty(size, self.num_heads, self.head_dim, dtype=cache["dtype"], device=cache["device"])
        new_v = torch.empty_like(new_k)
        old_len = cache["lengths"][block_idx]
        if k_buf is not None and old_len > 0:
            new_k[:old_len].copy_(k_buf[:old_len])
            new_v[:old_len].copy_(cache["v"][block_idx][:old_len])
        cache["k"][block_idx] = new_k
        cache["v"][block_idx] = new_v
        return new_k, new_v

    def _attention_kv_start(self, cache, block_idx, video_len_new):
        """First buffer row of the attention window (same keep rule as the cache)."""
        old_len = cache["lengths"][block_idx]
        if int(cache["window_size"]) <= 0:
            return 0
        video_len = min(old_len + video_len_new, cache["capacity"])
        return old_len + video_len_new - video_len

    def _commit_video_kv(self, cache, block_idx, video_len_new):
        """Make the current video K/V (already at rows [old_len, old_len + n)) part of the cache."""
        old_len = cache["lengths"][block_idx]
        n = video_len_new
        if int(cache["window_size"]) <= 0:
            cache["lengths"][block_idx] = old_len + n
            return
        capacity = cache["capacity"]
        k_buf, v_buf = cache["k"][block_idx], cache["v"][block_idx]
        if n >= capacity:
            src = slice(old_len + n - capacity, old_len + n)
            k_buf[:capacity].copy_(k_buf[src].clone())
            v_buf[:capacity].copy_(v_buf[src].clone())
            cache["lengths"][block_idx] = capacity
            return
        keep = min(old_len, capacity - n)
        if keep != old_len:
            src = slice(old_len - keep, old_len + n)
            k_buf[: keep + n].copy_(k_buf[src].clone())
            v_buf[: keep + n].copy_(v_buf[src].clone())
        cache["lengths"][block_idx] = keep + n

    def get_cross_attn_kv_cache(self, cache_name):
        cache = self.cross_attn_kv_caches.get(cache_name)
        if cache is None:
            cache = [None for _ in range(self.blocks_num)]
            self.cross_attn_kv_caches[cache_name] = cache
        return cache

    @staticmethod
    def _cross_attn_cache_key(cache_name, context):
        return (cache_name, context.data_ptr(), tuple(context.shape), str(context.device), str(context.dtype))

    def clear_cache(self, cache_name=None):
        if cache_name is None:
            self.kv_caches.clear()
            self.cross_attn_kv_caches.clear()
        else:
            cache_names = {cache_name, f"{cache_name}_cond", f"{cache_name}_uncond"}
            for name in cache_names:
                self.kv_caches.pop(name, None)
            for key in list(self.cross_attn_kv_caches.keys()):
                key_cache_name = key[0] if isinstance(key, tuple) else key
                if key_cache_name in cache_names:
                    self.cross_attn_kv_caches.pop(key, None)
        self._cu_seqlens_cache.clear()

    def _split_modulation(self, modulation, time_segments):
        """Per-segment AdaLN rows: ``table + embed0`` on the unique timestep rows only."""
        embed0 = time_segments.embed0
        table = modulation.tensor.to(embed0.device, embed0.dtype)
        values = table + embed0
        return [_SegModulation(item.reshape(-1, item.shape[-1]), time_segments.seg_idx) for item in values.chunk(6, dim=1)]

    def _get_cu_seqlens(self, length, device):
        key = (int(length), str(device))
        cached = self._cu_seqlens_cache.get(key)
        if cached is None:
            cached = torch.tensor([0, int(length)], device=device, dtype=torch.int32)
            self._cu_seqlens_cache[key] = cached
        return cached

    def _modulate(self, x, scale, shift):
        if _can_use_cuda_kernels(x) and x.is_contiguous():
            if self.is_thor and self.modulate_func is fuse_scale_shift_kernel and can_use_seg_kernels(x):
                return seg_modulate(x, scale.seg_idx, scale.rows, shift.rows)
            scale_arg = scale.tokens()
            shift_arg = shift.tokens()
            if x.dim() == 2:
                scale_arg = scale_arg.unsqueeze(0)
                shift_arg = shift_arg.unsqueeze(0)
            return self.modulate_func(x, scale=scale_arg, shift=shift_arg).reshape_as(x)
        return (x.float() * (1.0 + scale.tokens().float()) + shift.tokens().float()).to(x.dtype)

    def _gate_residual_(self, x, y, gate):
        """In place ``x += y * gate`` (BF16 rounding of the product kept)."""
        if self.is_thor and _can_use_cuda_kernels(x) and can_use_seg_kernels(x) and y.dtype == x.dtype and y.is_contiguous() and gate.rows.dtype == x.dtype:
            return seg_gate_residual_(x, y, gate.seg_idx, gate.rows)
        return x.add_(y * gate.tokens())

    def _fused_ln_params(self, norm, x, affine):
        """(weight, bias) if ``norm`` can run in the fused LayerNorm kernel, else None."""
        if not (self.fuse_layer_norm and type(norm) is LN_WEIGHT_REGISTER["torch"] and _can_use_cuda_kernels(x)):
            return None
        if norm.sensitive_layer_dtype != norm.infer_dtype or self.sensitive_layer_dtype != self.infer_dtype:
            return None
        if not can_use_torch_layer_norm_5120(x):
            return None
        weight, bias = norm._get_actual_weight(), norm._get_actual_bias()
        if not affine:
            return (None, None) if weight is None and bias is None else None
        for t in (weight, bias):
            if t is None or not (t.device == x.device and t.dtype == x.dtype and t.shape == (x.shape[1],) and t.is_contiguous() and t.data_ptr() % 16 == 0):
                return None
        return weight, bias

    def _apply_residual_(self, x, residual, gate):
        if residual is None:
            return x
        if gate is None:
            return x.add_(residual)
        return self._gate_residual_(x, residual, gate)

    def _norm_modulate(self, norm, x, scale, shift, residual=None, gate=None):
        """``modulate(norm(x'))``; x' = x + residual (* gate) is written back to x."""
        if self.modulate_func is fuse_scale_shift_kernel and self._fused_ln_params(norm, x, affine=False) is not None and (residual is None or residual.is_contiguous()):
            return fused_layer_norm(x, norm.eps, residual=residual, gate=None if gate is None else gate.rows, seg_idx=scale.seg_idx, scale=scale.rows, shift=shift.rows)
        self._apply_residual_(x, residual, gate)
        out = norm.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = out.to(self.sensitive_layer_dtype)
        out = self._modulate(out.contiguous(), scale, shift)
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = out.to(self.infer_dtype)
        return out

    def _norm_affine(self, norm, x, residual=None, gate=None):
        """``norm(x')`` with x' = x + residual (* gate) written back to x."""
        params = self._fused_ln_params(norm, x, affine=True)
        if params is not None and (residual is None or residual.is_contiguous()):
            return fused_layer_norm(x, norm.eps, weight=params[0], bias=params[1], residual=residual, gate=None if gate is None else gate.rows, seg_idx=None if gate is None else gate.seg_idx)
        self._apply_residual_(x, residual, gate)
        return norm.apply(x)

    def _fused_rms_norm_ok(self, norm, x):
        if not (self.fuse_rms_norm and type(norm) is RMSWeight and _can_use_cuda_kernels(x) and norm.sensitive_layer_dtype == norm.infer_dtype):
            return False
        return can_use_torch_rms_norm_5120(x, norm.weight, getattr(norm, "weight_diff", None))

    def _apply_attention_rms_norm(self, norm, x, out=None, rope_cos_sin=None):
        """Fuse torch RMSNorm before splitting the full hidden dimension into heads.

        Match the supported torch CUDA reduction and BF16 rounding exactly.
        Other shapes, torch versions, and norm backends keep their original path.
        If ``out`` is given the result is written there.  ``rope_cos_sin`` also
        applies the (flashinfer-identical) interleaved RoPE; callers must check
        ``_fused_rms_norm_ok`` before passing it.
        """
        if self._fused_rms_norm_ok(norm, x):
            return torch_rms_norm_5120(x, norm.weight, getattr(norm, "weight_diff", None), norm.eps, out=out, rope_cos_sin=rope_cos_sin)
        assert rope_cos_sin is None
        result = norm.apply(x)
        if out is not None:
            out.copy_(result)
            return out
        return result

    @staticmethod
    def _fused_rope_cos_sin(phase, pre_infer_out, q):
        """The fp32 [L, 128] cos|sin table if RoPE can be fused into RMSNorm."""
        rope = getattr(phase, "dreamzero_rope", None)
        freqs = pre_infer_out.freqs
        if type(rope) is not DreamZeroRope or not (_can_use_cuda_kernels(q) and rope.flashinfer_rope.is_available()):
            return None
        # DreamZeroRope.prepare_positions is arange(L): position == row.
        if not (torch.is_tensor(freqs) and freqs.dtype == torch.float32 and freqs.is_contiguous() and freqs.shape == (q.shape[0], 128) and freqs.device == q.device):
            return None
        return freqs

    def _linear(self, module, x):
        """``module.apply(x)``; on Thor the bitwise-identical CUTLASS kernel."""
        if self.is_thor:
            from lightx2v.common.ops.mm.thor_qkv import linear

            return linear(module, x)
        return module.apply(x)

    def infer_self_attn_with_cache(self, phase, x, shift_msa, scale_msa, pre_infer_out, kv_cache, pending_residual=None):
        """``pending_residual`` = (y, gate) of the previous block's FFN, applied
        to x (in place) fused with norm1."""
        seq_total = x.shape[0]
        n, d = self.num_heads, self.head_dim
        residual, gate = pending_residual if pending_residual is not None else (None, None)
        norm1_out = self._norm_modulate(phase.norm1, x, scale_msa, shift_msa, residual=residual, gate=gate)

        # Current K/V go straight into the cache buffers right after the cached
        # rows: [video tokens | action/state tokens].
        block_idx = self.block_idx
        old_len = kv_cache["lengths"][block_idx]
        action_len = pre_infer_out.action_register_length or 0
        video_len_new = seq_total - action_len
        k_buf, v_buf = self._kv_buffers(kv_cache, block_idx, old_len + seq_total)
        k_cur = k_buf[old_len : old_len + seq_total]
        v_cur = v_buf[old_len : old_len + seq_total]

        if not self.is_thor:
            q = phase.self_attn_q.apply(norm1_out)
            k = phase.self_attn_k.apply(norm1_out)
            v_cur.view(seq_total, n * d).copy_(phase.self_attn_v.apply(norm1_out))
        else:
            from lightx2v.common.ops.mm.thor_qkv import project_qkv

            q, k, _ = project_qkv(
                (phase.self_attn_q, phase.self_attn_k, phase.self_attn_v),
                norm1_out,
                outputs=(None, None, v_cur.view(seq_total, n * d)),
            )
        # q/k RMSNorm (+ RoPE when it can be fused); k is written into the cache.
        cos_sin = self._fused_rope_cos_sin(phase, pre_infer_out, q)
        if cos_sin is not None and self._fused_rms_norm_ok(phase.self_attn_norm_q, q) and self._fused_rms_norm_ok(phase.self_attn_norm_k, k):
            q = self._apply_attention_rms_norm(phase.self_attn_norm_q, q, rope_cos_sin=cos_sin).view(seq_total, n, d)
            self._apply_attention_rms_norm(phase.self_attn_norm_k, k, out=k_cur.view(seq_total, n * d), rope_cos_sin=cos_sin)
        else:
            q = self._apply_attention_rms_norm(phase.self_attn_norm_q, q).view(seq_total, n, d)
            k = self._apply_attention_rms_norm(phase.self_attn_norm_k, k, out=k_cur.view(seq_total, n * d)).view(seq_total, n, d)
            q, k = phase.dreamzero_rope.apply(
                q,
                k,
                pre_infer_out.freqs,
                positions=pre_infer_out.rope_positions,
            )
            if k.data_ptr() != k_cur.data_ptr():  # RoPE backend that is not in-place
                k_cur.copy_(k.view_as(k_cur))

        # Same window as the cache keep rule: the newest cached rows, then the
        # current video tokens, then the action/state registers.
        kv_start = self._attention_kv_start(kv_cache, block_idx, video_len_new)
        k_attn = k_buf[kv_start : old_len + seq_total]
        v_attn = v_buf[kv_start : old_len + seq_total]

        cu_q = self._get_cu_seqlens(q.shape[0], q.device)
        cu_kv = self._get_cu_seqlens(k_attn.shape[0], k_attn.device)
        attn_out = phase.self_attn_1.apply(
            q=q,
            k=k_attn,
            v=v_attn,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k_attn.shape[0],
        )
        if pre_infer_out.update_cache:
            self._commit_video_kv(kv_cache, block_idx, video_len_new)
        return self._linear(phase.self_attn_o, attn_out)

    def infer_cross_attn_dreamzero(self, phase, x, y_out, gate_msa, pre_infer_out):
        norm3_out = self._norm_affine(phase.norm3, x, residual=y_out, gate=gate_msa)
        context = pre_infer_out.context
        if context is None:
            raise ValueError("DreamZero cross-attention requires projected context.")

        n, d = self.num_heads, self.head_dim
        q = self._apply_attention_rms_norm(phase.cross_attn_norm_q, self._linear(phase.cross_attn_q, norm3_out)).view(-1, n, d)
        cross_attn_kv_cache = self.get_cross_attn_kv_cache(self._cross_attn_cache_key(pre_infer_out.cache_name, context))
        cached_kv = cross_attn_kv_cache[self.block_idx]
        if cached_kv is None:
            context_img = context[:257]
            context_txt = context[257:]
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context_txt)).view(-1, n, d)
            v = phase.cross_attn_v.apply(context_txt).view(-1, n, d)
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)
            cross_attn_kv_cache[self.block_idx] = (k, v, k_img, v_img)
        else:
            k, v, k_img, v_img = cached_kv

        cu_q = self._get_cu_seqlens(q.shape[0], q.device)
        cu_kv = self._get_cu_seqlens(k.shape[0], k.device)
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
        )

        cu_kv_img = self._get_cu_seqlens(k_img.shape[0], k_img.device)
        img_attn_out = phase.cross_attn_2.apply(
            q=q,
            k=k_img,
            v=v_img,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv_img,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k_img.shape[0],
        )
        attn_out.add_(img_attn_out)
        return x, self._linear(phase.cross_attn_o, attn_out)

    def infer_ffn_dreamzero(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        norm2_out = self._norm_modulate(phase.norm2, x, c_scale_msa, c_shift_msa, residual=attn_out)
        if self.is_thor:
            from lightx2v.common.ops.mm.thor_ffn.backend import up_gelu

            y = up_gelu(phase.ffn_0, norm2_out)
        else:
            y = phase.ffn_0.apply(norm2_out)
            y = F.gelu(y, approximate="tanh")
        if self.is_thor:
            from lightx2v.common.ops.mm.thor_ffn.backend import down_bias

            return down_bias(phase.ffn_2, y)
        return phase.ffn_2.apply(y)

    def _infer_block_deferred(self, block, x, pre_infer_out, kv_cache, pending_residual=None):
        """Run one block; its final ``x += y * c_gate`` is returned as pending so
        the next block can fuse it with its norm1."""
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self._split_modulation(
            block.compute_phases[0].modulation,
            pre_infer_out.time_segments,
        )
        y_out = self.infer_self_attn_with_cache(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
            pre_infer_out,
            kv_cache,
            pending_residual=pending_residual,
        )
        x, attn_out = self.infer_cross_attn_dreamzero(
            block.compute_phases[1],
            x,
            y_out,
            gate_msa,
            pre_infer_out,
        )
        y = self.infer_ffn_dreamzero(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        return x, (y, c_gate_msa)

    def infer_block(self, block, x, pre_infer_out, kv_cache):
        x, (y, c_gate_msa) = self._infer_block_deferred(block, x, pre_infer_out, kv_cache)
        self._gate_residual_(x, y, c_gate_msa)
        return x

    def infer_main_blocks(self, blocks, pre_infer_out, kv_cache):
        x = pre_infer_out.x
        pending = None
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x, pending = self._infer_block_deferred(blocks[block_idx], x, pre_infer_out, kv_cache, pending_residual=pending)
        if pending is not None:
            self._gate_residual_(x, *pending)
        return x

    def infer_action_decoder(self, weights, x, pre_infer_out):
        if pre_infer_out.action_length == 0:
            return None
        action_tokens = x[pre_infer_out.seq_len : pre_infer_out.seq_len + pre_infer_out.action_length].unsqueeze(0)
        hidden = F.relu(_category_linear(action_tokens, weights.action_decoder.layer1))
        return _category_linear(hidden, weights.action_decoder.layer2)

    def infer_video_head(self, weights, x, pre_infer_out):
        x = x[: pre_infer_out.seq_len]
        segments = pre_infer_out.time_segments
        seg_idx = segments.seg_idx[: pre_infer_out.seq_len]
        table = weights.head_modulation.tensor.to(x.device, x.dtype)
        shift, scale = [_SegModulation(item.reshape(-1, item.shape[-1]), seg_idx) for item in (table + segments.embed[:, None, :]).chunk(2, dim=1)]
        x = weights.norm.apply(x).to(x.dtype)
        x = self._modulate(x.contiguous(), scale, shift)
        x = weights.head.apply(x)
        return self.unpatchify(x, pre_infer_out.grid_size)

    def unpatchify(self, x, grid_size):
        batch_size = 1
        c = self.config["out_dim"]
        p_t, p_h, p_w = tuple(self.config.get("patch_size", (1, 2, 2)))
        f, h, w = grid_size
        if x.shape[0] != math.prod(grid_size):
            raise ValueError(f"DreamZero unpatchify expected {math.prod(grid_size)} tokens, got {x.shape[0]}.")
        x = x.view(batch_size, f, h, w, p_t, p_h, p_w, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(batch_size, c, f * p_t, h * p_h, w * p_w)

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        kv_cache = self.get_kv_cache(pre_infer_out.cache_name, dtype=pre_infer_out.x.dtype, device=pre_infer_out.x.device)
        x = self.infer_main_blocks(weights.blocks, pre_infer_out, kv_cache)
        video_noise_pred = self.infer_video_head(weights, x, pre_infer_out).to(GET_DTYPE())
        action_noise_pred = self.infer_action_decoder(weights, x, pre_infer_out)
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.to(GET_DTYPE())
        return video_noise_pred, action_noise_pred
