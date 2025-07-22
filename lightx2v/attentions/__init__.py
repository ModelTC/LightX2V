from lightx2v.attentions.common.torch_sdpa import torch_sdpa
from lightx2v.attentions.common.flash_attn2 import flash_attn2
from lightx2v.attentions.common.flash_attn3 import flash_attn3
from lightx2v.attentions.common.sage_attn2 import sage_attn2
from lightx2v.attentions.common.radial_attn import radial_attn


def attention(attention_type="flash_attn2", *args, **kwargs):
    if attention_type == "torch_sdpa":
        return torch_sdpa(*args, **kwargs)
    elif attention_type == "flash_attn2":
        return flash_attn2(*args, **kwargs)
    elif attention_type == "flash_attn3":
        return flash_attn3(*args, **kwargs)
    elif attention_type == "sage_attn2":
        return sage_attn2(*args, **kwargs)
    elif attention_type == "radial_attn":
        return radial_attn(*args, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported attention mode: {attention_type}")
