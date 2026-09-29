import torch

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER


@PLATFORM_ROPE_REGISTER("supa_rope")
class SupaQwenImageRope(RopeTemplate):
    def apply(self, xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache):
        import torch_br

        cos, sin = cos_sin_cache
        xq = xq.transpose(0, 1).contiguous()
        xk = xk.transpose(0, 1).contiguous()
        return torch_br.supa_rope_apply(xq, cos, sin), torch_br.supa_rope_apply(xk, cos, sin)

    @staticmethod
    def prepare_cos_sin_from_complex(rotary_emb):
        cos_half = rotary_emb.real.float()
        sin_half = rotary_emb.imag.float()
        cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0)
        sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0)
        return cos, sin
