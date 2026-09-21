import torch
import torch.distributed as dist
from torch.nn import functional as F

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER

try:
    import torch_br
except ImportError:
    torch_br = None


@PLATFORM_ROPE_REGISTER("biren_wan_rope")
class BirenWanRope(RopeTemplate):
    def __init__(self, layout="interleaved", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "interleaved":
            raise ValueError("BirenWanRope only supports interleaved layout.")
        self._positions = None
        self._cos_cache = None
        self._sin_cache = None
        self._cached_seq_len = -1
        self._cached_rank = -1
        self._cached_world_size = -1

    @staticmethod
    def prepare_cos_sin(cos_sin, grid_sizes, head_size, seq_p_group=None):
        c = head_size // 2
        f, h, w = grid_sizes
        seq_len = f * h * w
        cos_cat = cos_sin.real
        sin_cat = cos_sin.imag
        freqs_cat = torch.stack([cos_cat, sin_cat], dim=-1).reshape(seq_len, c, 2)
        if seq_p_group is not None:
            world_size = dist.get_world_size(seq_p_group)
            cur_rank = dist.get_rank(seq_p_group)
            seqlen = freqs_cat.shape[0]
            padding_size = ((seqlen - 1) // world_size + 1) * world_size - seqlen
            if padding_size > 0:
                freqs_cat = F.pad(freqs_cat, (0, 0, 0, 0, 0, padding_size), value=1)
            s_pad = freqs_cat.shape[0]
            cos_supa, sin_supa = [
                x.squeeze(2).reshape(s_pad, head_size)
                for x in freqs_cat.unsqueeze(-1).expand(s_pad, c, 2, 2).chunk(2, dim=2)
            ]
            local_len = s_pad // world_size
            start = cur_rank * local_len
            end = (cur_rank + 1) * local_len
            return torch.stack([sin_supa[start:end], cos_supa[start:end]], dim=0)

        cos_supa, sin_supa = [
            x.squeeze(2).reshape(seq_len, head_size)
            for x in freqs_cat.unsqueeze(-1).expand(seq_len, c, 2, 2).chunk(2, dim=2)
        ]
        return torch.stack([sin_supa, cos_supa], dim=0)

    @staticmethod
    def prepare_cos_sin_from_freqs(freqs, grid_sizes, head_size, seq_p_group=None):
        c = head_size // 2
        f, h, w = grid_sizes
        seq_len = f * h * w
        parts_cos = [
            freqs[0][:f].real.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].real.view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].real.view(1, 1, w, -1).expand(f, h, w, -1),
        ]
        parts_sin = [
            freqs[0][:f].imag.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].imag.view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].imag.view(1, 1, w, -1).expand(f, h, w, -1),
        ]
        cos_cat = torch.cat(parts_cos, dim=-1)
        sin_cat = torch.cat(parts_sin, dim=-1)
        freqs_cat = torch.stack([cos_cat, sin_cat], dim=-1).reshape(seq_len, c, 2)
        if seq_p_group is not None:
            world_size = dist.get_world_size(seq_p_group)
            cur_rank = dist.get_rank(seq_p_group)
            seqlen = freqs_cat.shape[0]
            padding_size = ((seqlen - 1) // world_size + 1) * world_size - seqlen
            if padding_size > 0:
                freqs_cat = F.pad(freqs_cat, (0, 0, 0, 0, 0, padding_size), value=1)
            s_pad = freqs_cat.shape[0]
            cos_supa, sin_supa = [
                x.squeeze(2).reshape(s_pad, head_size)
                for x in freqs_cat.unsqueeze(-1).expand(s_pad, c, 2, 2).chunk(2, dim=2)
            ]
            local_len = s_pad // world_size
            start = cur_rank * local_len
            end = (cur_rank + 1) * local_len
            return torch.stack([sin_supa[start:end], cos_supa[start:end]], dim=0)

        cos_supa, sin_supa = [
            x.squeeze(2).reshape(seq_len, head_size)
            for x in freqs_cat.unsqueeze(-1).expand(seq_len, c, 2, 2).chunk(2, dim=2)
        ]
        return torch.stack([sin_supa, cos_supa], dim=0)

    def _get_seq_parallel_rank(self):
        if not self.config.get("seq_parallel", False) or not dist.is_initialized():
            return 0
        device_mesh = self.config.get("device_mesh")
        if device_mesh is None:
            return 0
        return dist.get_rank(device_mesh.get_group(mesh_dim="seq_p"))

    def _get_seq_parallel_world_size(self):
        if not self.config.get("seq_parallel", False) or not dist.is_initialized():
            return 1
        device_mesh = self.config.get("device_mesh")
        if device_mesh is None:
            return 1
        return dist.get_world_size(device_mesh.get_group(mesh_dim="seq_p"))

    def apply(self, xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache: torch.Tensor):
        """
        Apply WAN RoPE using torch_br.supa_rope_infer_v2.

        Args:
            xq: Query tensor [seq_len, num_heads, head_dim]
            xk: Key tensor [seq_len, num_heads, head_dim]
            cos_sin_cache: Real tensor [seq_len, 1, head_dim//2, 2]
                           [..., 0] = cos, [..., 1] = sin

        Returns:
            Tuple of (xq, xk) with rotary embedding applied
        """
        batched_qkv = xq.dim() == 3 and xq.shape[0] == 1 and xq.shape[-1] == self.config["dim"]
        if batched_qkv:
            seq_len = xq.shape[1]
            num_heads = self.config["num_heads"]
            head_dim = self.config["dim"] // num_heads
        else:
            seq_len = xq.shape[0]
            num_heads = xq.shape[1]
            head_dim = xq.shape[2]

        seq_rank = self._get_seq_parallel_rank()
        world_size = self._get_seq_parallel_world_size()
        if (
            self._cached_seq_len != seq_len
            or self._cached_rank != seq_rank
            or self._cached_world_size != world_size
        ):
            self._cached_seq_len = seq_len
            self._cached_rank = seq_rank
            self._cached_world_size = world_size
            if cos_sin_cache.dim() == 3 and cos_sin_cache.shape[0] == 2:
                cache_seq_len = cos_sin_cache.shape[1]
            else:
                cache_seq_len = cos_sin_cache.shape[0]
            if cache_seq_len != seq_len:
                raise ValueError(f"RoPE cache seq_len mismatch: cache={cache_seq_len}, q={seq_len}")

            if cos_sin_cache.dim() == 3 and cos_sin_cache.shape[0] == 2:
                self._sin_cache = cos_sin_cache[0].contiguous()
                self._cos_cache = cos_sin_cache[1].contiguous()
            else:
                # Compatibility path for the old [seq_len, 1, head_dim//2, 2] cache.
                cos_half = cos_sin_cache[:, 0, :, 0]
                sin_half = cos_sin_cache[:, 0, :, 1]
                self._cos_cache = cos_half.unsqueeze(-1).expand(-1, -1, 2).reshape(seq_len, head_dim).contiguous()
                self._sin_cache = sin_half.unsqueeze(-1).expand(-1, -1, 2).reshape(seq_len, head_dim).contiguous()
            self._positions = torch.arange(0, seq_len * world_size, 1, dtype=torch.int32, device=xq.device)

        # Golden Wan2.2 keeps TP-2Die q/k as [1, seq_len, num_heads * head_dim]
        # through rope and only then applies shape_transpose_pre_attn.
        if batched_qkv:
            q = xq
            k = xk
        else:
            q = xq.unsqueeze(0).reshape(1, seq_len, num_heads * head_dim)
            k = xk.unsqueeze(0).reshape(1, seq_len, num_heads * head_dim)

        q, k = torch_br.supa_rope_infer_v2(
            q,
            k,
            self._sin_cache,
            self._cos_cache,
            self._positions,
            head_dim,
            rope_type="Wan",
        )

        if not batched_qkv:
            q = q.reshape(seq_len, num_heads, head_dim)
            k = k.reshape(seq_len, num_heads, head_dim)

        return q, k
